import re
from dataclasses import dataclass
from typing import Any, List, Dict, Union

import torch
from zhconv import convert

import torch
import numpy as np
import matplotlib.pyplot as plt

import string


def get_part_of_dataset(dataset,ratio):
    num_samples=int(ratio*len(dataset))
    return dataset[:num_samples]

def generate_random_string(length):
    # 获取所有可能的字符集合
    all_chars = string.ascii_letters + string.digits
    all_chars=list(all_chars)
    # 确保长度不超过字符集合的大小
    # length = min(length, len(all_chars))

    # 从字符集合中随机选择字符直到达到指定的长度
    random_string = ''.join(np.random.choice(all_chars, length))

    return random_string


def random_prob(low_prob=0.2, high_prob=0.8):
    # 生成0到1之间的随机数
    rand_num = torch.rand(1).item()

    # 对随机数进行线性变换
    prob = low_prob + rand_num * (high_prob - low_prob)

    return prob


def random_discrete_only_mask(signal_shape, unit=(1, 40), prob=0.5):
    # signal: [64, 2000] tensor
    # unit:[1,40]
    length = int(np.ceil(signal_shape[1] / unit[1]))
    channel_num = int(np.ceil(signal_shape[0] / unit[0]))
    pre_mask = torch.rand(channel_num, length)
    pre_mask[pre_mask >= prob] = 1
    pre_mask[pre_mask < prob] = 0
    pre_mask = torch.repeat_interleave(pre_mask, int(np.ceil(signal_shape[0] / channel_num)), dim=0)
    pre_mask = torch.repeat_interleave(pre_mask, int(np.ceil(signal_shape[1] / length)), dim=1)[:signal_shape[0],
               :signal_shape[1]]
    return pre_mask


def random_discrete_mask(signal, unit=(1, 40), prob=0.5):
    # signal: [64, 2000] tensor
    # unit:[1,40]
    length = int(np.ceil(signal.shape[1] / unit[1]))
    channel_num = int(np.ceil(signal.shape[0] / unit[0]))
    pre_mask = torch.rand(channel_num, length)
    pre_mask[pre_mask >= prob] = 1
    pre_mask[pre_mask < prob] = 0
    pre_mask = torch.repeat_interleave(pre_mask, int(np.ceil(signal.shape[0] / channel_num)), dim=0)
    pre_mask = torch.repeat_interleave(pre_mask, int(np.ceil(signal.shape[1] / length)), dim=1)[:signal.shape[0],
               :signal.shape[1]]
    return pre_mask


def random_channel_mask(signal, low=1, high=32):
    # 随机选择掩码通道数量
    mask_size = torch.randint(low, high + 1, (1,)).item()

    # 随机选择要掩码的通道
    channels = torch.randperm(signal.shape[0])[:mask_size]
    mask = torch.ones_like(signal)
    # 对选择的通道进行掩码
    mask[channels, :] = 0

    return mask


def random_length_mask(signal, unit_length=40, low_prob=0.2, high_prob=0.8):
    prob = random_prob(low_prob, high_prob)
    length = int(np.ceil(signal.shape[1] / unit_length))
    pre_mask = torch.rand(1, length)
    pre_mask[pre_mask >= prob] = 1
    pre_mask[pre_mask < prob] = 0
    pre_mask = pre_mask.repeat_interleave(signal.shape[0], axis=0)
    pre_mask = pre_mask.repeat_interleave(unit_length, axis=1)
    pre_mask = pre_mask[:, :signal.shape[1]]
    return signal * pre_mask, pre_mask


class RandomShapeMasker:
    def __init__(self, unit=(1, 40), mask_prob=0.7, channel_num=(1, 32), length_prob=(0.2, 0.4), random_types=(1,)):
        self.unit = unit
        self.mask_prob = mask_prob
        self.channel_num = channel_num
        self.length_prob = length_prob
        self.random_types = random_types

    def __call__(self, signal_shape):
        random_type = np.random.choice(self.random_types, 1)[0]
        if random_type == 1:
            return random_discrete_only_mask(signal_shape, unit=self.unit, prob=self.mask_prob)


# 删除标点符号
def remove_punctuation(text: str or List[str]):
    punctuation = '!,.;:?、！，。；：？'
    if isinstance(text, str):
        text = re.sub(r'[{}]+'.format(punctuation), '', text).strip()
        return text
    elif isinstance(text, list):
        result_text = []
        for t in text:
            t = re.sub(r'[{}]+'.format(punctuation), '', t).strip()
            result_text.append(t)
        return result_text
    else:
        raise Exception(f'不支持该类型{type(text)}')


# 将繁体中文总成简体中文
def to_simple(text: str or List[str]):
    if isinstance(text, str):
        text = convert(text, 'zh-cn')
        return text
    elif isinstance(text, list):
        result_text = []
        for t in text:
            t = convert(t, 'zh-cn')
            result_text.append(t)
        return result_text
    else:
        raise Exception(f'不支持该类型{type(text)}')



def contains_valid_letters(s, prefix='Ġ',biaodian=',.\'`:?'):
    if len(s)<1:
        return False
    if prefix==s[0]:
        s=s[1:]
    return re.match(f"^[A-Za-z{biaodian}]+$", s) is not None



@dataclass
class DataCollatorOnlySpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        # input_features = [{"input_features": torch.tensor(feature["input_features"][0])} for feature in features]
        # for feature in input_features:
        #     print(feature["input_features"].shape)
        batch = {
            'input_features': torch.stack(
                [torch.tensor(feature["input_features"][0], dtype=torch.float32) for feature in features])
        }
        # print(features[0].keys())
        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        # input_features = [{"input_features": torch.tensor(feature["input_features"][0])} for feature in features]
        # for feature in input_features:
        #     print(feature["input_features"].shape)
        batch={
            'input_features':torch.stack([torch.tensor(feature["input_features"][0],dtype=torch.float32) for feature in features])
        }
        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        vocab_size=51865
        input_ids=labels
        if torch.max(input_ids) > vocab_size:
            print('input_ids bigger than vocab')
            print(torch.max(input_ids))
            print(f'input_ids:{input_ids}')
            print(f'input_ids shape:{input_ids.shape}')
            # 检查 input_ids 中超过词表大小的元素
            input_ids = input_ids.reshape(-1)
            exceed_indices = torch.where(input_ids >= vocab_size)
            exceed_sequences = input_ids[exceed_indices]
            print(f"超过词表大小的input_ids:{exceed_sequences}")
            exceed_count = (input_ids > vocab_size).sum().item()
            print(f"超过词表大小的元素数量: {exceed_count},占比:{exceed_count / input_ids.numel()}")
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
