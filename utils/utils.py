import hashlib
import os
import tarfile
import urllib.request

from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import RobustScaler

import scipy.signal

def preprocess_eeg_data(data,threshold = 10):

    # 1. Baseline correction
    mean_baseline = data[:,:500].mean(axis=1)
    data = data - mean_baseline[:,None]
    # 2. Robust scaling
    scaler = RobustScaler()
    data = scaler.fit_transform(data.T).T

    # 3. Clipping outliers

    data[np.abs(data) > threshold] = np.sign(data[np.abs(data) > threshold]) * threshold
    data=data/threshold
    threshold_mask = np.abs(data) > 1
    num_clipped = np.sum(threshold_mask)

    # 计算比例
    clipped_ratio = num_clipped / (data.shape[0]*data.shape[1])
    return data,clipped_ratio


def add_gaussian_noise(signal_input, snr_range):
    # 获取信号的形状和通道数
    ch, length = signal_input.shape

    # 生成每个通道的信噪比
    snr_per_channel = np.random.uniform(*snr_range, size=ch)

    # 初始化噪声信号
    noise_signal = np.zeros_like(signal_input)

    # 逐通道添加高斯噪声
    for i in range(ch):
        # 计算当前通道的信噪比
        snr = snr_per_channel[i]

        # 计算当前通道的噪声标准差
        noise_std = np.sqrt(np.mean(signal_input[i] ** 2) / (10 ** (snr / 10)))

        # 生成高斯噪声
        noise = np.random.normal(scale=noise_std, size=length)

        # 添加噪声到当前通道
        noise_signal[i] = signal_input[i] + noise

    # 应用噪声
    noisy_signal = signal_input + noise_signal

    return noisy_signal

def lowpass_filter(signal_input, cutoff_freq, sample_freq):
    # 计算归一化的截止频率
    normalized_cutoff_freq = cutoff_freq / (sample_freq / 2)

    # 设计低通滤波器
    b, a = scipy.signal.butter(4, normalized_cutoff_freq, btype='low', analog=False, output='ba')

    # 应用滤波器
    filtered_signal = scipy.signal.lfilter(b, a, signal_input, axis=0)

    return filtered_signal

def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in vars(args).items():
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def strtobool(val):
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


def str_none(val):
    if val == 'None':
        return None
    else:
        return val


def add_arguments(argname, type, default, help, argparser, **kwargs):
    type = strtobool if type == bool else type
    type = str_none if type == str else type
    argparser.add_argument("--" + argname,
                           default=default,
                           type=type,
                           help=help + ' Default: %(default)s.',
                           **kwargs)


def md5file(fname):
    hash_md5 = hashlib.md5()
    f = open(fname, "rb")
    for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
    f.close()
    return hash_md5.hexdigest()


def download(url, md5sum, target_dir):
    """Download file from url to target_dir, and check md5sum."""
    if not os.path.exists(target_dir): os.makedirs(target_dir)
    filepath = os.path.join(target_dir, url.split("/")[-1])
    if not (os.path.exists(filepath) and md5file(filepath) == md5sum):
        print(f"Downloading {url} to {filepath} ...")
        with urllib.request.urlopen(url) as source, open(filepath, "wb") as output:
            with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True,
                      unit_divisor=1024) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    loop.update(len(buffer))
        print(f"\nMD5 Chesksum {filepath} ...")
        if not md5file(filepath) == md5sum:
            raise RuntimeError("MD5 checksum failed.")
    else:
        print(f"File exists, skip downloading. ({filepath})")
    return filepath


def unpack(filepath, target_dir, rm_tar=False):
    """Unpack the file to the target_dir."""
    print("Unpacking %s ..." % filepath)
    tar = tarfile.open(filepath)
    tar.extractall(target_dir)
    tar.close()
    if rm_tar:
        os.remove(filepath)


def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)
