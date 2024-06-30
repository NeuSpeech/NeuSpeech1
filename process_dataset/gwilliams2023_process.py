import mne
import numpy as np
import mne.io
import soundfile as sf
import pandas as pd
from sklearn.preprocessing import RobustScaler
import jsonlines
import os
import librosa
from IPython.display import Audio,display
import tqdm


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)  # 设置 Python 内置随机库的种子
    np.random.seed(seed)  # 设置 NumPy 随机库的种子
    torch.manual_seed(seed)  # 设置 PyTorch 随机库的种子
    torch.cuda.manual_seed(seed)  # 为当前 CUDA 设备设置种子
    torch.cuda.manual_seed_all(seed)  # 为所有 CUDA 设备设置种子


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
def get_sequences(tsv_path):
    text = pd.read_csv(tsv_path, delimiter='\t')
    words_dict = []
    for i in range(len(text)):
        # print(text.iloc[i])
        tti = eval(text['trial_type'][i])
        if tti['kind'] == 'word':
            words_dict.append({'onset': text.iloc[i]['onset'], 'duration': text.iloc[i]['duration'], **tti})

    sentences = []
    old_sequence_id = None
    seq_count = 0

    for word in words_dict:
        sequence_id = int(word['sequence_id'])

        if sequence_id != old_sequence_id:
            sentences.append({'words': []})
            seq_count += 1

        sentences[seq_count - 1]['words'].append(word)

        old_sequence_id = sequence_id

    # 更新句子的起始时间和结束时间
    for sequence_id, sentence_info in enumerate(sentences):
        sentences[sequence_id]['story'] = sentences[sequence_id]['words'][0]['story']
        sentences[sequence_id]['story_id'] = sentences[sequence_id]['words'][0]['story_uid']
        sentences[sequence_id]['sound_id'] = sentences[sequence_id]['words'][0]['sound_id']
        sentences[sequence_id]['seq_id'] = sentences[sequence_id]['words'][0]['sequence_id']
        sentences[sequence_id]['speech_rate'] = sentences[sequence_id]['words'][0]['speech_rate']
        sentences[sequence_id]['voice'] = sentences[sequence_id]['words'][0]['voice']
        sentences[sequence_id]['meg_path'] = tsv_path[:-10] + 'meg.con'
        sentences[sequence_id]['audio_path'] = sentences[sequence_id]['words'][0]['sound']
        sentences[sequence_id]['start'] = sentences[sequence_id]['words'][0]['onset']
        sentences[sequence_id]['end'] = sentences[sequence_id]['words'][-1]['onset'] + \
                                        sentences[sequence_id]['words'][-1]['duration']
        sentences[sequence_id]['audio_start'] = sentences[sequence_id]['words'][0]['start']
        sentences[sequence_id]['audio_end'] = sentences[sequence_id]['words'][-1]['start'] + \
                                              sentences[sequence_id]['words'][-1]['duration']
        sentences[sequence_id]['duration'] = sentences[sequence_id]['audio_end'] - sentences[sequence_id]['audio_start']
        sentences[sequence_id]['text'] = ' '.join([word['word'] for word in sentences[sequence_id]['words']])
    return sentences


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


def find_files_with_extension(folder_path, extension):
    file_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.abspath(os.path.join(root, file))
                file_paths.append(file_path)

    return file_paths


def write_jsonlines(file_path, json_dicts):
    with jsonlines.open(file_path, mode='w') as writer:
        for json_dict in json_dicts:
            writer.write(json_dict)


def makedirs(path):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    return path


# 按照句子的时间去切割
def process_meg(tsv_path):
    print(tsv_path)
    target_meg_sr = 200
    sentences = get_sequences(tsv_path)
    meg_path = sentences[0]['meg_path']
    meg = mne.io.read_raw_kit(meg_path, preload=True, verbose=False)
    picks = mne.pick_types(
        meg.info, meg=True,ref_meg=True, eeg=False, stim=False, eog=False, ecg=False
    )
    meg.pick(picks, verbose=False)
    meg.filter(l_freq=1, h_freq=58, verbose=False)
    meg.resample(target_meg_sr)
    data = meg.get_data()
    assert data.shape[0]==224, f'data shape:{data.shape}'
    speech_sr = None
    target_speech_sr = 16000
    old_audio_path = None
    lines = []
    for i, sent in enumerate(sentences):
        # 切分meg
        start_meg_index = int(sent['start'] * target_meg_sr)
        end_meg_index = int(sent['end'] * target_meg_sr)
        seg_meg = data[:, start_meg_index:end_meg_index]
        if detect_outliers(seg_meg)!='正常':
            break
        # 切分音频
        audio_path = sent['audio_path']
        if old_audio_path != audio_path:
            speech_data, speech_sr = sf.read(os.path.join(folder_path, audio_path.lower()))
        start_audio_index = int(sent['audio_start'] * speech_sr)
        end_audio_index = int(sent['audio_end'] * speech_sr)
        seg_audio = speech_data[start_audio_index:end_audio_index]
        # 标准化
        seg_meg, cr = preprocess_eeg_data(seg_meg, threshold=10)
        # 重采样
        seg_audio = librosa.resample(seg_audio, orig_sr=speech_sr, target_sr=target_speech_sr)
        # 将处理好的音频文件，脑电文件，标注文件都储存好。
        seg_meg_path = tsv_path.replace('download', replace_folder).replace('events.tsv', f'senid_{i}_meg.npy')
        seg_audio_path = seg_meg_path.replace('meg.npy', 'audio.wav')
        if detect_outliers(seg_meg)!='正常':
            break
        makedirs(seg_meg_path)
        # print(f'{i} seg_meg {seg_meg.shape} seg_meg_path {seg_meg_path}')
        np.save(seg_meg_path, seg_meg)
        # seg_meg = np.load(seg_meg_path)
        # print('load seg_meg',seg_meg.shape,seg_meg_path)
        sf.write(seg_audio_path, seg_audio, target_speech_sr)

        # 解析其他的键值对
        selected_keys = ['story', 'story_id', 'seq_id', 'sound_id', 'speech_rate', 'voice', 'start', 'end',
                         'audio_start', 'audio_end']

        new_dict = {key: sent[key] for key in selected_keys}
        # 做 whisper json
        line = {
            "speech": {"path": seg_audio_path, 'sr': target_speech_sr},
            "eeg": {"path": seg_meg_path, 'sr': target_meg_sr},
            "duration": sent['duration'],
            "language": "English",
            "sentence": sent['text'],
            "sentences": [{"text": sent['text'],
                           "start": 0.0, "end": sent['duration'], "duration": sent['duration'],
                           "words": [{"word": word['word'], "start": word['onset'] - sent['audio_start'],
                                      "end": word['onset'] + word['duration'] - sent['audio_start']} for word in
                                     sent['words']]}],
            "subj":int(os.path.basename(tsv_path)[4:6]),
            **new_dict
        }
        #         selected_keys = ['duration',"sentence",'story', 'story_id', 'seq_id','sound_id', 'speech_rate', 'voice','start','end','audio_start','audio_end']

        #         new_dict = {key: line[key] for key in selected_keys}
        # print(new_dict)
        # display(Audio(line["speech"]["path"]))
        lines.append(line)
    seg_jsonl_path = tsv_path.replace('download', replace_folder).replace('events.tsv', 'info.jsonl')
    write_jsonlines(seg_jsonl_path, lines)
    # print(tsv_path,'done')
    return lines


from multiprocessing import Pool


def read_jsonlines(file_path):
    json_dicts = []
    with jsonlines.open(file_path, mode='r') as reader:
        for json_obj in reader:
            json_dicts.append(json_obj)
    return json_dicts

def process_file(filename):
    lines = process_meg(filename)
    return lines

def detect_outliers_json(json_dict):
    meg=np.load(json_dict['eeg']['path'])
    do=detect_outliers(meg)
    if do!='正常':
        print(do,json_dict)
    return do
if __name__ == '__main__':
    replace_folder = 'preprocess5'
    folder_path = '/hpc2hdd/home/yyang937/datasets/gwilliams2023/download/'
    extension = 'events.tsv'
    events_tsv_list = find_files_with_extension(folder_path, extension)
    set_random_seed(seed=42)
    pool = Pool(processes=32)
    results = pool.map(process_file, events_tsv_list)
    pool.close()
    pool.join()

    all_lines = []
    for lines in results:
        all_lines.extend(lines)
    print("检查异常值")
    pool = Pool(processes=32)
    results = pool.map(detect_outliers_json, all_lines)
    pool.close()
    pool.join()
    print(set(results))
    print("检查完毕")
    write_jsonlines(os.path.join(folder_path.replace('download', replace_folder), 'info.jsonl'), all_lines)

    # 将数据分为训练集，验证集和测试集。
    # 第一种分法是将所有数据直接随机分为8:1:1
    # 第二种分法是将session1的数据分为训练和验证9:1，测试集为整个session2的数据。
    import random

    data = read_jsonlines(f'/hpc2hdd/home/yyang937/datasets/gwilliams2023/{replace_folder}/info.jsonl')  # 替换为你的数据列表


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
    split1_path = os.path.join(folder_path.replace('download', replace_folder), 'split1')
    os.makedirs(split1_path, exist_ok=True)
    write_jsonlines(os.path.join(split1_path, 'train.jsonl'), train_data1)
    write_jsonlines(os.path.join(split1_path, 'val.jsonl'), val_data1)
    write_jsonlines(os.path.join(split1_path, 'test.jsonl'), test_data1)
