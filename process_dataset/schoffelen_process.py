# 音频与脑电对齐
import pandas as pd
import mne
import numpy as np
import jsonlines
from sklearn.preprocessing import RobustScaler,StandardScaler
import librosa
import soundfile as sf
import os
import re
import time
import tqdm
import random
from multiprocessing import Pool

def extract_string(string):
    if type(string)!=str:
        return None
    pattern =r'\d+(\D+)\d+'
    match = re.search(pattern, string)
    if match:
        output= match.group(1).strip()
        if output == '':
            return None
        return output
    else:
        return None
def get_stimuli_dict():
    file_path = "/hpc2hdd/home/yyang937/datasets/schoffelen2019n/DSC_3011020.09_236_v1/stimuli/stimuli.txt"

    data_dict = {}

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            key, value = line.split(" ", 1)
            data_dict[key] = value
    return data_dict

def detect_outliers(arr):
    # 判断超过20%的值为0
    if np.count_nonzero(arr == 0) / len(arr) > 0.2:
        return f"{np.count_nonzero(arr == 0) / len(arr)*100}%的值为0"

    # 判断是否存在NaN或None
    if np.shape(arr)[1]==0:
        return "数组长度为0"

    # 判断是否存在NaN或None
    if np.isnan(arr).any() or None in arr:
        return "存在NaN或None"

    # 判断是否存在正负无穷大
    if np.isinf(arr).any():
        return "存在正负无穷大"

    # 可以添加其他异常情况的判断逻辑

    return "正常"

def read_tsv_auditory(tsv_path,data_dict):
    # 读取TSV文件
    df = pd.read_csv(tsv_path, delimiter='\t')

    # 创建空列表来存储结果
    result = []

    # 初始化变量来追踪最近的type
    prev_type = None

    # 遍历每一行数据
    for index, row in df.iterrows():
        # 更新prev_type变量
        if 'ZINNEN' == row['value']:
            prev_type = 'ZINNEN'
        elif 'WOORDEN' == row['value']:
            prev_type = 'WOORDEN'
        # 检查是否为音频开始行
        if row['type'] == 'Sound' and row['value'].endswith('.wav'):
            # 提取音频编号和onset值
            speech = row['value'][-7:-4]
            onset = int(row['sample'])

            # 确定type字段
            if prev_type == 'ZINNEN':
                type_ = 'ZINNEN'
            elif prev_type == 'WOORDEN':
                type_ = 'WOORDEN'
            else:
                type_ = 'unknown'

            # 创建字典并添加到结果列表中
            result.append({
                'audio_path': f'/hpc2hdd/home/yyang937/datasets/schoffelen2019n/DSC_3011020.09_236_v1/stimuli/audio_files/EQ_Ramp_Int2_Int1LPF{speech}.wav',
                'text':data_dict[str(int(speech))],
                'onset': onset,
                'type': type_,
                'meg_path': tsv_path.replace('events.tsv','meg.ds'),
                    'stimuli_type':'audio'
            })
        if row['type']=='Nothing' and 'End of file' in row['value']:
            result[-1]['offset']=row['sample']
    return result


def preprocess_eeg_data(data, threshold=10):
    data=data.T

    # 2. Robust scaling
    scaler = RobustScaler()
    scaler.fit(data[:100])
    data = scaler.transform(data).T

    # 3. Clipping outliers

    data[np.abs(data) > threshold] = np.sign(data[np.abs(data) > threshold]) * threshold
    data = data / threshold
    threshold_mask = np.abs(data) > 1
    num_clipped = np.sum(threshold_mask)

    # 计算比例
    clipped_ratio = num_clipped / (data.shape[0] * data.shape[1])
    assert clipped_ratio<0.2,'clip ratio should below 20%'
    return data, clipped_ratio


def write_jsonlines(file_path, json_dicts):
    with jsonlines.open(file_path, mode='w') as writer:
        for json_dict in json_dicts:
            writer.write(json_dict)


def makedirs(path):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    return path


# 按照句子的时间去切割
def process_audio_meg(tsv_path,data_dict):
    target_meg_sr = 200
    target_speech_sr = 16000
    print(tsv_path)
    sentences = read_tsv_auditory(tsv_path,data_dict=data_dict)
    meg_path = sentences[0]['meg_path']
    try:
        meg = mne.io.read_raw_ctf(meg_path, preload=True, verbose=False)
    except Exception as e:
        return []
    picks = mne.pick_types(
        meg.info, meg=True, eeg=False, stim=False, eog=False, ecg=False
    )[: (28 + 273)]  # 从meta学的，不要ref通道
    meg.pick(picks, verbose=False)
    meg.notch_filter(50, verbose=False)
    meg.filter(l_freq=1, h_freq=60, verbose=False)
    meg_sr = meg.info['sfreq']
    meg.resample(target_meg_sr)
    data = meg.get_data()

    lines = []
    # print(f'len sentences:{len(sentences)}')
    for i, sent in tqdm.tqdm(enumerate(sentences)):
        audio_path = sent['audio_path']
        speech_data, speech_sr = sf.read(audio_path)
        if speech_data.shape[1]==2:
            speech_data=speech_data[:,0]
        # 切分meg
        # data = data[:224]
        start_meg_index = int(sent['onset']/meg_sr*target_meg_sr)
        end_meg_index = int(sent['offset']/meg_sr*target_meg_sr)
        seg_meg = data[:, start_meg_index:end_meg_index]
        duration=(end_meg_index-start_meg_index)/target_meg_sr
        doo=detect_outliers(seg_meg)
        if doo != "正常":
            print(f'tsv_path{tsv_path},i:{i},sent:{sent},data.shape:{data.shape},'
                  f'smi:{start_meg_index},emi:{end_meg_index},duration:{duration}')
            # print(i)
            # print(sent)
            # print(data.shape,start_meg_index,end_meg_index,duration)
            break
        seg_audio = librosa.resample(speech_data, orig_sr=speech_sr, target_sr=target_speech_sr)
        # 标准化
        seg_meg, cr = preprocess_eeg_data(seg_meg, threshold=10)
        # 将处理好的音频文件，脑电文件，标注文件都储存好。
        seg_meg_path = tsv_path.replace(mid_folder, replace_folder).replace('events.tsv', f'senid_{i}_meg.npy')
        seg_audio_path = seg_meg_path.replace('meg.npy', 'audio.wav')
        makedirs(seg_meg_path)
        np.save(seg_meg_path, seg_meg)
        sf.write(seg_audio_path, seg_audio, target_speech_sr)

        # 做 whisper json
        line = {
            "speech": {"path": os.path.abspath(seg_audio_path), 'sr': target_speech_sr},
            "eeg": {"path": os.path.abspath(seg_meg_path), 'sr': target_meg_sr},
            "duration": duration,
            "language": "Dutch",
            "sentence_id": audio_path[-7:-4],
            "sentence": sent['text'],
            "sentences": [{"text": sent['text'],
                           "start": 0.0, "end": duration, "duration": duration
                           }],
            'subj':os.path.basename(tsv_path)[5:9],
            'stimuli_type':'audio',
            'sent_type':sent['type']
        }
        lines.append(line)

    seg_jsonl_path = tsv_path.replace(mid_folder, replace_folder).replace('_events.tsv', f'.jsonl')
    write_jsonlines(seg_jsonl_path, lines)
    # print(tsv_path,'done')
    return lines

def read_tsv_visual(tsv_path):
    # 读取TSV文件
    df = pd.read_csv(tsv_path, delimiter='\t')

    # 创建空列表来存储结果
    result = []

    # 初始化变量来追踪最近的type
    prev_type = None

    # 遍历每一行数据
    for index, row in df.iterrows():
        # 更新prev_type变量
        if 'ZINNEN' == row['value']:
            prev_type = 'ZINNEN'
        elif 'WOORDEN' == row['value']:
            prev_type = 'WOORDEN'
        # 检查是否为音频开始行
        value_ext=extract_string(row['value'])
        if row['type'] == 'Picture' :
            if value_ext is not None:
                # 提取音频编号和onset值
                onset = int(row['sample'])

                # 确定type字段
                if prev_type == 'ZINNEN':
                    type_ = 'ZINNEN'
                elif prev_type == 'WOORDEN':
                    type_ = 'WOORDEN'
                else:
                    type_ = 'unknown'

                # 创建字典并添加到结果列表中
                result.append({
                    'text':value_ext,
                    'onset': onset,
                    'type': type_,
                    'meg_path': tsv_path.replace('events.tsv','meg.ds'),
                    'stimuli_type':'visual'
                })
            if row['value']=='ISI':
                result[-1]['offset']=row['sample']

    return result


def process_visual_meg(tsv_path):
    sentences = read_tsv_visual(tsv_path)
    meg_path = sentences[0]['meg_path']
    meg = mne.io.read_raw_ctf(meg_path, preload=True, verbose=False)
    picks = mne.pick_types(
        meg.info, meg=True, eeg=False, stim=False, eog=False, ecg=False
    )[28: (28 + 273)]  # 从meta学的，不要ref通道
    meg.pick(picks, verbose=False)
    meg.notch_filter(50, verbose=False)
    meg.filter(l_freq=1, h_freq=60, verbose=False)
    data = meg.get_data()
    meg_sr = meg.info['sfreq']
    target_meg_sr = 200
    lines = []
    for i, sent in enumerate(sentences):
        # 切分meg
        # data = data[:224]
        start_meg_index = int(sent['onset'])
        end_meg_index = int(sent['offset'])
        seg_meg = data[:, start_meg_index:end_meg_index]
        duration=(end_meg_index-start_meg_index)/meg_sr
        # 切分音频 这里的音频不用切分
        # 标准化
        try:
            seg_meg, cr = preprocess_eeg_data(seg_meg, threshold=10)
        except Exception as e:
            continue
        # 重采样
        seg_meg = librosa.resample(seg_meg, orig_sr=meg_sr, target_sr=target_meg_sr)
        # 将处理好的音频文件，脑电文件，标注文件都储存好。
        seg_meg_path = tsv_path.replace(mid_folder, replace_folder).replace('events.tsv', f'senid_{i}_meg.npy')
        makedirs(seg_meg_path)
        np.save(seg_meg_path, seg_meg)

        # 做 whisper json
        line = {
            "speech": {"path": None, 'sr': None},
            "eeg": {"path": os.path.abspath(seg_meg_path), 'sr': target_meg_sr},
            "duration": duration,
            "sentence": sent['text'],
            "sentences": [{"text": sent['text'],
                           "start": 0.0, "end": duration, "duration": duration
                           }],
            'stimuli_type':'visual',
            'subj':os.path.basename(tsv_path)[4:9],
            'sent_type':sent['type']
        }
        lines.append(line)
    seg_jsonl_path = tsv_path.replace(mid_folder, replace_folder).replace('_events.tsv', f'.jsonl')
    write_jsonlines(seg_jsonl_path, lines)
    # print(tsv_path,'done')
    return lines




def process_audio_file(filename):
    try:
        lines = process_audio_meg(filename,data_dict=data_dict)
    except Exception as e:
        print(filename,e)
        lines = []
    return lines

def process_visual_file(filename):
    try:
        lines = process_visual_meg(filename)
    except Exception as e:
        print(filename,e)
        lines = []
    return lines
def find_files_with_extension(folder_path, extension):
    file_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.abspath(os.path.join(root, file))
                file_paths.append(file_path)

    return file_paths


def read_jsonlines(file_path):
    json_dicts = []
    with jsonlines.open(file_path, mode='r') as reader:
        for json_obj in reader:
            json_dicts.append(json_obj)
    return json_dicts


if __name__ == '__main__':
    data_dict = get_stimuli_dict()
    replace_folder = 'preprocess6'
    mid_folder = 'DSC_3011020.09_236_v1'
    # subj_audio_list=range(2002,2125)
    folder_path='/hpc2hdd/home/yyang937/datasets/schoffelen2019n'
    extension = 'events.tsv'
    set_random_seed(seed=42)
    events_tsv_list = find_files_with_extension(folder_path, extension)
    audio_events_tsv_list=[]
    visual_events_tsv_list=[]
    for i,tsv_path in enumerate(events_tsv_list):
        task_type=os.path.basename(tsv_path)
        if 'rest' in task_type or 'run' in task_type:
            continue
        dirname=os.path.dirname(tsv_path)
        dir_type=os.path.basename(dirname) # meg
        if dir_type!='meg':
            continue
        mode=os.path.basename(os.path.dirname(dirname))[-5]
        if mode=='V':
            visual_events_tsv_list.append(tsv_path)
        elif mode=='A':
            audio_events_tsv_list.append(tsv_path)

    # subj_audio_list=range(2004,2016)
    # events_tsv_list = [f'datasets/schoffelen2019n/DSC_3011020.09_236_v1/sub-A{subj_idx}/meg/sub-A{subj_idx}_task-auditory_events.tsv' for subj_idx in subj_audio_list]

    pool = Pool(processes=32)
    results = pool.map(process_audio_file, audio_events_tsv_list)
    pool.close()
    pool.join()

    all_lines1 = []
    for lines in results:
        all_lines1.extend(lines)

    write_jsonlines(os.path.abspath(f'/hpc2hdd/home/yyang937/datasets/schoffelen2019n/{replace_folder}/audio_info.jsonl'), all_lines1)

    # visual
    # subj_audio_list=range(1001,1118)
    # subj_audio_list=range(1011,1123)
    # events_tsv_list = [f'datasets/schoffelen2019n/DSC_3011020.09_236_v1/sub-V{subj_idx}/meg/sub-V{subj_idx}_task-visual_events.tsv' for subj_idx in subj_audio_list]
    # 避免在做标准化的时候出现0长度的数据。
    # pool = Pool(processes=12)
    # results = pool.map(process_visual_file, visual_events_tsv_list)
    # pool.close()
    # pool.join()
    #
    # all_lines2 = []
    # for lines in results:
    #     all_lines2.extend(lines)
    # write_jsonlines(os.path.abspath(f'datasets/schoffelen2019n/{replace_folder}/visual_info.jsonl'), all_lines2)
    # all_lines=all_lines1+all_lines2
    #
    # write_jsonlines(os.path.abspath(f'datasets/schoffelen2019n/{replace_folder}/all_info.jsonl'), all_lines)

    data = read_jsonlines(f'/hpc2hdd/home/yyang937/datasets/schoffelen2019n/{replace_folder}/info.jsonl')  # 替换为你的数据列表

    random.shuffle(data)  # 随机打乱数据列表

    total_samples = len(data)
    train_samples = int(0.8 * total_samples)
    val_samples = int(0.1 * total_samples)
    test_samples = total_samples - train_samples - val_samples

    train_data1 = data[:train_samples]
    val_data1 = data[train_samples:train_samples + val_samples]
    test_data1 = data[train_samples + val_samples:]

    print("训练集大小:", len(train_data1))
    print("验证集大小:", len(val_data1))
    print("测试集大小:", len(test_data1))
    split1_path = os.path.join(f'/hpc2hdd/home/yyang937/datasets/schoffelen2019n/{replace_folder}', 'split1')
    os.makedirs(split1_path, exist_ok=True)
    write_jsonlines(os.path.join(split1_path, 'train.jsonl'), train_data1)
    write_jsonlines(os.path.join(split1_path, 'val.jsonl'), val_data1)
    write_jsonlines(os.path.join(split1_path, 'test.jsonl'), test_data1)


