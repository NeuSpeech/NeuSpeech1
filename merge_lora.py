import argparse
import functools
import os
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizerFast,\
    WhisperProcessor

from utils.load_model import WhisperForConditionalGeneration
from peft import PeftModel, PeftConfig
from utils.utils import print_arguments, add_arguments
from utils.model_utils import projection_module

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("lora_model", type=str, default="output/whisper-tiny/checkpoint-final/", help="微调保存的模型路径")
add_arg("model_path", type=str, default="openai/whisper-base", help="base model")
add_arg("eeg_ch", type=int, default=0, help="通道的维度")
add_arg("local_files_only", type=bool, default=True, help="是否只在本地加载模型，不尝试下载")
args = parser.parse_args()
print_arguments(args)
# 获取Lora配置参数
peft_config = PeftConfig.from_pretrained(args.lora_model)
# 检查模型文件是否存在
assert os.path.exists(args.lora_model), f"模型文件{args.lora_model}不存在"
model = WhisperForConditionalGeneration.from_pretrained(args.model_path,
                                                    device_map="auto",
                                                    local_files_only=args.local_files_only,)
feature_extractor = WhisperFeatureExtractor.from_pretrained(peft_config.base_model_name_or_path,
                                                            local_files_only=args.local_files_only)
tokenizer = WhisperTokenizerFast.from_pretrained(peft_config.base_model_name_or_path,
                                                 local_files_only=args.local_files_only)
processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path,
                                             local_files_only=args.local_files_only)
device=model.device
kwargs={
    'meg_ch':args.eeg_ch,
    'd_model':model.model.encoder.conv2.in_channels,
}

conv1=projection_module(config_name='base',**kwargs)
model.model.encoder.set_input_embeddings(conv1)
if args.lora_model is not None:
    model = PeftModel.from_pretrained(model, args.lora_model, local_files_only=args.local_files_only)
    model = model.merge_and_unload()
model.train(False)

# 保存的文件夹路径
save_directory = os.path.join(args.lora_model, f'full_model')
os.makedirs(save_directory, exist_ok=True)

# 保存模型到指定目录中
model.save_pretrained(save_directory)
feature_extractor.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
processor.save_pretrained(save_directory)
print(f'合并模型保持在：{save_directory}')

