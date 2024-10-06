import json
import os
import sys
from typing import List
from utils.augment_eeg import RandomShapeMasker, shift_data
import librosa
import numpy as np
import soundfile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.utils import preprocess_eeg_data, lowpass_filter, add_gaussian_noise
import jsonlines
import re
import copy
import soundfile as sf


def torch_random_choices(samples, choices):
    indices = torch.randperm(len(samples))[:choices]
    rand_choices = [samples[i] for i in indices]
    return rand_choices


def read_jsonlines(file_path):
    json_dicts = []
    with jsonlines.open(file_path, mode='r') as reader:
        for json_obj in reader:
            json_dicts.append(json_obj)
    return json_dicts


def write_jsonlines(file_path, json_dicts):
    with jsonlines.open(file_path, mode='w') as writer:
        for json_dict in json_dicts:
            writer.write(json_dict)


def filter_ascii_str(text):
    return re.sub(r'[^a-zA-Z ]', '', text)


def filter_ascii_data_dict(data_dict):
    data_dict['sentence']=filter_ascii_str(data_dict['sentence'])
    for i,sentence in enumerate(data_dict['sentences']):
        sentence['text']=filter_ascii_str(sentence['text'])
        if "words" in sentence.keys():
            for j,w in enumerate(sentence['words']):
                w['word']=filter_ascii_str(w['word'])
    return data_dict


class SpeechDataset(Dataset):
    def __init__(self,
                 data_list_path,
                 processor,
                 data_list_dir='/home/yyang/dataset/multi_media/',
                 level='sentences',
                 language=None,
                 timestamps=False,
                 min_duration=0.5,
                 max_duration=30,):
        """
        Args:
            data_list_path: 数据列表文件的路径，或者二进制列表的头文件路径
            processor: Whisper的预处理工具，WhisperProcessor.from_pretrained获取
            modal: eeg,speech
            language: 微调数据的语言
            timestamps: 微调时是否使用时间戳
            sample_rate: 音频的采样率，默认是16000
            min_duration: 小于这个时间段的音频将被截断，单位秒，不能小于0.5，默认0.5s
            max_duration: 大于这个时间段的音频将被截断，单位秒，不能大于30，默认30s
            augment_config_path: 数据增强配置参数文件路径
        """
        super().__init__()
        assert min_duration >= 0.5, f"min_duration不能小于0.5，当前为：{min_duration}"
        assert max_duration <= 30, f"max_duration不能大于30，当前为：{max_duration}"
        self.data_list_path = data_list_path

        self.processor = processor
        self.language = language
        self.timestamps = timestamps
        self.data_list_dir = data_list_dir
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.vocab = self.processor.tokenizer.get_vocab()
        self.timestamp_begin = self.vocab['<|notimestamps|>'] + 1
        self.startoftranscript = self.vocab['<|startoftranscript|>']
        self.endoftext = self.vocab['<|endoftext|>']
        self.nocaptions = self.vocab['<|nocaptions|>']
        self.data_list: List[dict] = []
        # 加载数据列表
        self._load_data_list()

    # 加载数据列表
    def _load_data_list(self):
        data_list = read_jsonlines(self.data_list_path)  # [:1024] #
        self.data_list = data_list
        print(f'num of data:{len(self.data_list)}')

    # 从数据列表里面获取音频数据、采样率和文本
    def _get_list_data(self, idx):
        data_list = copy.deepcopy(self.data_list[idx])
        # 分割音频路径和标签
        # audio_file = os.path.join(self.data_list_dir, data_list[self.modal]["path"])
        audio_file = data_list['speech']["path"]
        transcript = data_list["sentences"] if self.timestamps else data_list["sentence"]
        language = data_list["language"] if 'language' in data_list.keys() else None
        sample, sample_rate = sf.read(audio_file, dtype='float32', always_2d=True)  # [len,ch]
        assert sample_rate == 16000, '输入的音频采样率应该是16kHz'

        sample = sample.T  # eeg:[ch, len]
        assert sample.shape[1] != 0
        return sample, sample_rate, transcript, language

    def __getitem__(self, idx):
        sample, sample_rate, transcript, language = self._get_list_data(idx=idx)
        self.processor.tokenizer.set_prefix_tokens(language=language if language is not None else self.language)
        data = self.processor(audio=sample,sampling_rate=sample_rate,text=transcript)
        return data

    def __len__(self):
        return len(self.data_list)



class CustomDataset(Dataset):
    def __init__(self,
                 data_list_path,
                 processor,
                 data_list_dir='/home/yyang/dataset/multi_media/',
                 mode='train',
                 modal='eeg',
                 modal_ch=66,
                 level='sentences',
                 language=None,
                 filter_dataset=False,
                 timestamps=False,
                 sample_rate=200,
                 orig_sample_rate=200,
                 min_duration=0.5,
                 max_duration=30,
                 combine_sentences=False,
                 split_sentences=False,
                 subj=None,
                 augment_config_path=None):
        """
        Args:
            data_list_path: 数据列表文件的路径，或者二进制列表的头文件路径
            processor: Whisper的预处理工具，WhisperProcessor.from_pretrained获取
            modal: eeg,speech
            language: 微调数据的语言
            timestamps: 微调时是否使用时间戳
            sample_rate: 音频的采样率，默认是16000
            min_duration: 小于这个时间段的音频将被截断，单位秒，不能小于0.5，默认0.5s
            max_duration: 大于这个时间段的音频将被截断，单位秒，不能大于30，默认30s
            augment_config_path: 数据增强配置参数文件路径
        """
        super(CustomDataset, self).__init__()
        assert min_duration >= 0.5, f"min_duration不能小于0.5，当前为：{min_duration}"
        assert max_duration <= 30, f"max_duration不能大于30，当前为：{max_duration}"
        self.data_list_path = data_list_path
        self.mode = mode
        self.level = level
        self.processor = processor
        self.signal_sample_rate = sample_rate
        self.orig_sample_rate = orig_sample_rate
        self.language = language
        self.filter_dataset = filter_dataset
        self.timestamps = timestamps
        self.combine_sentences = combine_sentences
        self.split_sentences = split_sentences
        self.data_list_dir = data_list_dir
        self.modal = modal
        self.modal_ch = modal_ch
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.subj = subj
        self.vocab = self.processor.tokenizer.get_vocab()
        self.timestamp_begin = self.vocab['<|notimestamps|>'] + 1
        self.startoftranscript = self.vocab['<|startoftranscript|>']
        self.endoftext = self.vocab['<|endoftext|>']
        self.nocaptions = self.vocab['<|nocaptions|>']
        self.data_list: List[dict] = []
        # 加载数据列表
        self._load_data_list()
        # 数据增强配置参数
        self.augment_configs = None
        self.noises_path = None
        self.speed_rates = None
        if augment_config_path:
            with open(augment_config_path, 'r', encoding='utf-8') as f:
                self.augment_configs = json.load(f)

        # self._scan_data()

    def _filter_schoffelen_sentence_data_list(self, data_list):
        # 过滤时间不符合限制的数据
        print(f'filtering')
        # self.data_list=[filter_ascii_data_dict(data) for data in self.data_list]
        # data_list=[]
        indices=[]
        # print(data_list[:2])
        for i,data in enumerate(data_list):
            if data["sent_type"]=="ZINNEN" and data["duration"]<30: # 处理数据的时候算错了，应该改成原来的采样率
                # print('schoffelen ZINNEN')
                indices.append(i)
                # data_list.append(data)
        return [data_list[i] for i in indices]

    def _filter_subj_data_list(self, data_list):
        # 过滤时间不符合限制的数据
        print(f'filtering')
        # self.data_list=[filter_ascii_data_dict(data) for data in self.data_list]
        # data_list=[]
        indices=[]
        # print(data_list[:2])
        for i,data in enumerate(data_list):
            if data["subj"]==self.subj: # 处理数据的时候算错了，应该改成原来的采样率
                # print('schoffelen ZINNEN')
                indices.append(i)
                # data_list.append(data)
        return [data_list[i] for i in indices]


    def _scan_data(self):
        for i in range(self.__len__()):
            self.__getitem__(i)


    # 加载数据列表
    def _load_data_list(self):
        # if self.mode.startswith('train'):
        #     # num=1000
        #     self.data_list = read_jsonlines(self.data_list_path)#[:num]
        # elif self.mode.startswith('val'):
        #     # num=10
        #     self.data_list = read_jsonlines(self.data_list_path)[500:600]
        # else:
        #     num=None
        #     self.data_list = read_jsonlines(self.data_list_path)[:num]
        data_list = read_jsonlines(self.data_list_path)#[:1024] #
        if self.filter_dataset:
            data_list=self._filter_schoffelen_sentence_data_list(data_list)
        if self.subj is not None:
            data_list=self._filter_subj_data_list(data_list)

        self.data_list=data_list
        print(f'num of data:{len(self.data_list)} mode:{self.mode}')

    # 从数据列表里面获取音频数据、采样率和文本
    def _get_list_data(self, idx):
        data_list = copy.deepcopy(self.data_list[idx])
        # 分割音频路径和标签
        # audio_file = os.path.join(self.data_list_dir, data_list[self.modal]["path"])
        audio_file = data_list[self.modal]["path"]
        # print(f'audio_file:{audio_file} self.data_list_dir{self.data_list_dir} data_list[self.modal]["path"]:{data_list[self.modal]["path"]}')
        assert audio_file is not None
        dataset_name=None
        if 'schoffelen' in audio_file:
            dataset_name='schoffelen'
        elif 'gwilliams' in audio_file:
            dataset_name='gwilliams'
        transcript = data_list["sentences"] if self.timestamps else data_list["sentence"]
        # transcript[0]['text']=f'{np.random.randint(0, 10)}'  #
        # transcript = transcript[:1]
        language = data_list["language"] if 'language' in data_list.keys() else None
        if self.modal == 'eeg':
            sample = np.load(audio_file)
            # assert sample.shape[0] < sample.shape[1], f'eeg shape should be [ch,len],now shape is {sample.shape},data idx{idx}'
            if dataset_name=='schoffelen': #todo 现在还没有想好怎么取一部分通道，就先这样特事特办了。
                sample = sample[28:301]
            elif dataset_name=='gwilliams':
                sample = sample[:208]
            else:
                sample = sample[:self.modal_ch]
            # 如果channel更多，就padding到指定多的通道数
            if self.modal_ch>sample.shape[0]:
                sample=self.pad_sample_ch(sample)
            sample = sample.T  # convert to shape[len,ch]
            sample_rate = self.orig_sample_rate
        elif self.modal == 'speech':
            sample, sample_rate = soundfile.read(audio_file, dtype='float32',always_2d=True)  # [len,ch]
            assert sample_rate==16000,'输入的音频采样率应该是16kHz'
            self.signal_sample_rate = sample_rate
            self.orig_sample_rate = sample_rate
        else:
            raise NotImplementedError

        sample = sample.T  # eeg:[ch, len]
        # print(sample.shape)
        if self.modal == 'eeg':
            sample_rate = self.signal_sample_rate
            # assert clipped_ratio < 0.2
        # 数据增强
        if self.augment_configs:
            # 只有训练的时候才会增强数据
            if self.mode == 'train':
                sample, sample_rate = self.augment_audio(sample, sample_rate)
        # 重采样

        return sample, sample_rate, transcript, language

    def _get_list_data_random_split(self,idx):
        sample, sample_rate, transcript, language=self._get_list_data(idx)
        ratio=np.random.rand()
        ratio=ratio*0.8+0.2
        words=transcript.split()
        seg_sample_length=int(sample.shape[1]*ratio)
        seg_transcript_length=max(int(len(words) *ratio),1)
        if np.random.rand()>0.5:# 截取后面的一段
            sample=sample[:,-seg_sample_length:]
            words=words[-seg_transcript_length:]
        else:
            sample=sample[:,:seg_sample_length]
            words=words[:seg_transcript_length]
        transcript=' '.join(words)
        return sample, sample_rate, transcript, language

    def _get_list_data_random(self,idx):
        # 这个会把sample的空间填满。每个sample之间有随机2到4s的间隔。
        # 现在只支持 无timestamps,单language
        assert self.timestamps is False
        max_allow_length=int(self.max_duration*self.signal_sample_rate)
        sample, sample_rate, transcript, language=self._get_list_data(idx)
        if np.random.rand()>0.5:
            ch,sample_length=sample.shape
            full_length=sample_length
            for i in range(3): # 最多加三句
                rand_sec=np.random.random()
                rand_length=int(rand_sec*self.signal_sample_rate)

                new_sample, sample_rate, new_transcript, language = self._get_list_data(np.random.randint(self.__len__()))

                if new_sample.shape[1]+rand_length+full_length<max_allow_length:
                    # print(sample.shape,[ch,rand_length],new_sample.shape)
                    sample=np.concatenate([sample,np.zeros([ch,rand_length]),new_sample],axis=1)
                    transcript=transcript+f'{"" if transcript.endswith(".") else "."}'+' '+new_transcript
                    # print(i,transcript)
                    full_length=full_length+rand_length+new_sample.shape[1]
        return sample, sample_rate, transcript, language




    def _load_timestamps_transcript(self, transcript: List[dict]):
        level = self.level
        if level == 'words':
            return self._load_timestamps_transcript_words(transcript)
        elif level == 'sentences':
            return self._load_timestamps_transcript_sentences(transcript)
        else:
            raise NotImplementedError

    def _load_timestamps_transcript_sentences(self, transcript: List[dict]):
        assert isinstance(transcript, list), f"transcript应该为list，当前为：{type(transcript)}"
        data = dict()
        labels = self.processor.tokenizer.prefix_tokens[:3]
        # print(f'transcript :{len(transcript),transcript}')
        for t in transcript:
            # 将目标文本编码为标签ID
            start = t['start'] if round(t['start'] * 100) % 2 == 0 else t['start'] + 0.01
            start = self.timestamp_begin + round(start * 100) // 2
            end = t['end'] if round(t['end'] * 100) % 2 == 0 else t['end'] - 0.01
            end = self.timestamp_begin + round(end * 100) // 2
            label = self.processor(text=t['text']).input_ids[4:-1]
            # print(f'len label:{len(label)} label:{label} transcript:{transcript}')
            if max(label)>51865:
                print(f'OOV text {t["text"]} label {label}\n')
                raise ValueError
            if start>51865:
                print(f'OOV start {t["start"]} label {start}\n')
                raise ValueError
            if end>51865:
                print(f'OOV start {t["end"]} label {end}\n')
                raise ValueError
            labels.extend([start])
            labels.extend(label)
            labels.extend([end])
        data['labels'] = labels + [self.endoftext]
        return data

    def _load_timestamps_transcript_words(self, transcript: List[dict]):
        assert isinstance(transcript, list), f"transcript应该为list，当前为：{type(transcript)}"
        data = dict()
        labels = self.processor.tokenizer.prefix_tokens[:3]
        for t in transcript:
            # 将目标文本编码为标签ID
            words = t['words']
            for w in words:
                start = w['start'] if round(w['start'] * 100) % 2 == 0 else w['start'] + 0.01
                start = self.timestamp_begin + round(start * 100) // 2
                end = w['end'] if round(w['end'] * 100) % 2 == 0 else w['end'] - 0.01
                end = self.timestamp_begin + round(end * 100) // 2
                label = self.processor(text=w['word']).input_ids[4:-1]
                labels.extend([start])
                labels.extend(label)
                labels.extend([end])
        data['labels'] = labels + [self.endoftext]
        return data

    def shift_data_transcript(self, sample, transcript):
        # assert self.modal == 'eeg'
        assert isinstance(transcript, list), f"transcript应该为list，当前为：{type(transcript)}"
        length =max(int((transcript[-1]["end"])*self.signal_sample_rate),sample.shape[1])  # notice 应该设置到最长的这个长度，来避免移出去。
        assert length / self.signal_sample_rate < self.max_duration, f'做数据漂移的时间长度必须小于{self.max_duration}s,现在的长度是{length}，采样率是{self.signal_sample_rate},时间长度是{length / self.signal_sample_rate}'
        # try:
        max_shift = int(self.max_duration * self.signal_sample_rate) - length -0.5*self.signal_sample_rate  # 留余量
        # except Exception as e :
        #     print(f'{length},{e}')
        now_shift = np.random.randint(max_shift, size=None)
        sample = shift_data(sample, now_shift)
        now_shift_time = now_shift / self.signal_sample_rate
        for t in transcript:
            # 将目标文本编码为标签ID
            old_start=t['start']
            old_end=t['end']
            t['start'] = t['start'] + now_shift_time
            t['end'] = t['end'] + now_shift_time

            if t['start']>=30 or t['end']>=30:
                print(f''
                      f'now_shift:{now_shift}\n'
                      f'max_shift:{max_shift}\n'
                      f'now_shift_time:{now_shift_time}\n'
                      f'self.signal_sample_rate:{self.signal_sample_rate}\n'
                      f'self.max_duration:{self.max_duration}\n'
                      f't old start:{old_start}\n'
                      f't old end:{old_end}\n'
                      f't start:{transcript[0]["start"]}\n'
                      f't end:{transcript[0]["end"]}\n'
                      f'self.max_duration:{self.max_duration}\n'
                      f'transcript:{transcript}\n'
                      f'')
                raise ValueError
            if self.level == 'words':
                for w in t['words']:
                    w['start'] = w['start'] + now_shift_time
                    w['end'] = w['end'] + now_shift_time
        return sample, transcript



    def __getitem__(self, idx):
        # try:
        # 从数据列表里面获取音频数据、采样率和文本
        if self.combine_sentences:
            sample, sample_rate, transcript, language = self._get_list_data_random(idx=idx)
        elif self.split_sentences:
            sample, sample_rate, transcript, language = self._get_list_data_random_split(idx=idx)
        else:
            sample, sample_rate, transcript, language = self._get_list_data(idx=idx)
        # transcript[0]=f'{np.random.choice([i for i in range(10)])}'
        # 将sample进行时间漂移，并将transcript的时间对齐。
        if self.mode == 'train':
            if torch.rand(1).item() < self.augment_configs['shift']['prob']:
                sample, transcript = self.shift_data_transcript(sample, transcript)
        # 可以为单独数据设置语言
        self.processor.tokenizer.set_prefix_tokens(language=language if language is not None else self.language)
        if len(transcript) > 0:
            # 加载带有时间戳的文本
            if self.timestamps:
                data = self._load_timestamps_transcript(transcript=transcript)
                if self.modal == 'speech':
                    data["input_features"] = self.processor(audio=sample,
                                                            sampling_rate=self.signal_sample_rate).input_features
                    # print(f'input_features:{data["input_features"][0].shape}')
                else:
                    data["input_features"] = self.padding_sample(sample)
            else:
                if self.modal == 'speech':
                    data = self.processor(audio=sample, sampling_rate=self.signal_sample_rate, text=transcript)
                else:
                    data = {
                        'input_features': self.padding_sample(sample),
                        'labels': self.process_transcript(transcript),
                    }

        else:
            # 如果没有文本，则使用<|nocaptions|>标记
            print('没有文本')
            if self.modal == 'speech':
                data = self.processor(audio=sample, sampling_rate=self.signal_sample_rate)
            else:
                data = {'input_features': self.padding_sample(sample),
                        'labels': [self.startoftranscript, self.nocaptions, self.endoftext]}
        # print(f'mode:{self.mode}   data:{idx}')
        # print(f'data {data["input_features"][0].shape}')
        return data

        # except Exception as e:
        #     print(f'读取数据出错，序号：{idx}，错误信息：{e}', file=sys.stderr)
        #     return self.__getitem__(torch.randint(0, self.__len__(),(1,)).item())

    def padding_sample(self, sample):
        assert self.modal == 'eeg'
        max_length = int(self.max_duration * self.signal_sample_rate)
        sample = sample[:, :max_length]
        # print(f'before pad eeg:{sample.shape}')
        sample = np.pad(sample, pad_width=((0, 0), (0, max_length - sample.shape[-1])))
        # print(f'sample.shape:{sample.shape}')
        assert sample.shape == (self.modal_ch,
                                int(self.signal_sample_rate * self.max_duration)), ' sample shape should be [self.modal_ch,int(self.signal_sample_rate*30))]'
        # print(f'after pad eeg:{sample.shape}')
        return [sample]

    def pad_sample_ch(self,sample):
        assert len(sample.shape)==2,f'sample.shape is now {sample.shape}'
        if sample.shape[0]==self.modal_ch:
            return sample
        # 现在就是说直接融合两个数据集。在后面添加通道
        assert sample.shape[0]<self.modal_ch,'sample channel must be less than modal channel'
        sample = np.pad(sample, pad_width=((0,  self.modal_ch - sample.shape[0]), (0,0)))
        assert sample.shape[0]==self.modal_ch,f'after pad ch, sample channel should be {self.modal_ch}, now is {sample.shape[0]}'
        return sample


    def process_transcript(self, transcript):
        data = self.processor(text=transcript)
        return data['input_ids']

    def __len__(self):
        return len(self.data_list)

    # 分割读取音频
    @staticmethod
    def slice_from_file(file, start, end):
        sndfile = soundfile.SoundFile(file)
        sample_rate = sndfile.samplerate
        duration = round(float(len(sndfile)) / sample_rate, 3)
        start = round(start, 3)
        end = round(end, 3)
        # 从末尾开始计
        if start < 0.0: start += duration
        if end < 0.0: end += duration
        # 保证数据不越界
        if start < 0.0: start = 0.0
        if end > duration: end = duration
        if end < 0.0:
            raise ValueError("切片结束位置(%f s)越界" % end)
        if start > end:
            raise ValueError("切片开始位置(%f s)晚于切片结束位置(%f s)" % (start, end))
        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)
        sndfile.seek(start_frame)
        sample = sndfile.read(frames=end_frame - start_frame, dtype='float32')
        return sample, sample_rate

    # 数据增强

    def augment_audio(self, sample, sample_rate):
        for k,v in self.augment_configs.items():
            # if config['type'] == 'volume' and torch.rand(1).tolist()[0] < config['prob']:
            #     min_gain_dBFS, max_gain_dBFS = config['params']['min_gain_dBFS'], config['params']['max_gain_dBFS']
            #     gain = torch.randint(min_gain_dBFS, max_gain_dBFS,(1,)).tolist()[0]
            #     sample = self.volume(sample, gain=gain)

            if k == 'noise' and torch.rand(1).item() < v['prob']:
                if self.modal == 'eeg':
                    sample = add_gaussian_noise(sample, snr_range=(
                    v['min_snr_dB'], v['max_snr_dB']))
            if k == 'mask' and torch.rand(1).item() < v['prob']:
                if self.modal == 'speech':
                    augmentor = RandomShapeMasker(unit=(None, None), mask_prob=None, length_unit=1600,
                                                  length_prob=(0.1, 0.2), channel_num=None, random_types=(3))
                    mask = augmentor(sample.shape)
                    del augmentor
                    mask = np.array(mask)
                    sample = sample * mask
                elif self.modal == 'eeg':
                    # eeg 目前是做椒盐噪声，即随机掩码
                    # print('eeg mask')
                    augmentor = RandomShapeMasker(**v['kwargs'])
                    mask = augmentor(sample.shape)
                    del augmentor
                    mask = np.array(mask)
                    sample = sample * mask
                else:
                    raise NotImplementedError
            if k == 'taylor' and torch.rand(1).item() < v['prob']:
                # 随机裁剪前后10个点的数据
                if self.modal == 'eeg':
                    mask=np.ones_like(sample)
                    # 把前后的随机10点的数据裁剪掉。
                    num=np.random.randint(1,10)
                    mask[:,:num]=0
                    num1=np.random.randint(1,10)
                    mask[:,-num1:]=0
                    # print(num,num1)
                    sample=sample*mask


        return sample, sample_rate

    # 改变语速
    @staticmethod
    def change_speed(sample, speed_rate):
        if speed_rate == 1.0:
            return sample
        if speed_rate <= 0:
            raise ValueError("速度速率应大于零")
        old_length = sample.shape[0]
        new_length = int(old_length / speed_rate)
        old_indices = np.arange(old_length)
        new_indices = np.linspace(start=0, stop=old_length, num=new_length)
        sample = np.interp(new_indices, old_indices, sample).astype(np.float32)
        return sample

    # 音频偏移
    @staticmethod
    def shift(sample, sample_rate, shift_ms):
        duration = sample.shape[0] / sample_rate
        if abs(shift_ms) / 1000.0 > duration:
            raise ValueError("shift_ms的绝对值应该小于音频持续时间")
        shift_samples = int(shift_ms * sample_rate / 1000)
        if shift_samples > 0:
            sample[:-shift_samples] = sample[shift_samples:]
            sample[-shift_samples:] = 0
        elif shift_samples < 0:
            sample[-shift_samples:] = sample[:shift_samples]
            sample[:-shift_samples] = 0
        return sample

    # 改变音量
    @staticmethod
    def volume(sample, gain):
        sample = sample * 10. ** (gain / 20.)
        return sample

    # 声音重采样
    @staticmethod
    def resample(sample, orig_sr, target_sr):
        sample = librosa.resample(sample, orig_sr=orig_sr, target_sr=target_sr)
        return sample

    # 添加噪声
    def add_noise(self, sample, sample_rate, noise_path, snr_dB, max_gain_db=300.0):
        noise_sample, sr = librosa.load(noise_path, sr=sample_rate)
        # 标准化音频音量，保证噪声不会太大
        target_db = -20
        gain = min(max_gain_db, target_db - self.rms_db(sample))
        sample = sample * 10. ** (gain / 20.)
        # 指定噪声音量
        sample_rms_db, noise_rms_db = self.rms_db(sample), self.rms_db(noise_sample)
        noise_gain_db = min(sample_rms_db - noise_rms_db - snr_dB, max_gain_db)
        noise_sample = noise_sample * 10. ** (noise_gain_db / 20.)
        # 固定噪声长度
        if noise_sample.shape[0] < sample.shape[0]:
            diff_duration = sample.shape[0] - noise_sample.shape[0]
            noise_sample = np.pad(noise_sample, (0, diff_duration), 'wrap')
        elif noise_sample.shape[0] > sample.shape[0]:
            start_frame = torch.randint(0, noise_sample.shape[0] - sample.shape[0], (1,)).item()
            noise_sample = noise_sample[start_frame:sample.shape[0] + start_frame]
        sample += noise_sample
        return sample

    @staticmethod
    def rms_db(sample):
        mean_square = np.mean(sample ** 2)
        return 10 * np.log10(mean_square)
