import argparse
import functools
import gc
import json
import os

import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor
from peft import PeftModel
from utils.data_utils import (DataCollatorSpeechSeq2SeqWithPadding,
                              remove_punctuation, DataCollatorOnlySpeechSeq2SeqWithPadding)
from utils.reader import SpeechDataset
from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("test_data",   type=str, default="dataset/test.json",            help="测试集的路径")
add_arg("base_model",  type=str, default="models/whisper-tiny-finetune", help="合并模型的路径，或者是huggingface上模型的名称")
add_arg("lora_model", type=str, default=None, help="训练过的lora模型")
add_arg("load_lora_model", type=bool, default=True, help="是否加载lora模型")
add_arg("modal", type=str, default='speech', help="输入的模态")
add_arg("batch_size",  type=int, default=16,        help="评估的batch size")
add_arg("num_workers", type=int, default=8,         help="读取数据的线程数量")
add_arg("language",    type=str, default="Chinese", help="设置语言，可全称也可简写，如果为None则评估的是多语言")
add_arg("remove_pun",  type=bool, default=True,     help="是否移除标点符号")
add_arg("to_simple",   type=bool, default=True,     help="是否转为简体中文")
add_arg("timestamps",  type=bool, default=False,    help="评估时是否使用时间戳数据")
add_arg("min_audio_len",     type=float, default=0.5,  help="最小的音频长度，单位秒")
add_arg("max_audio_len",     type=float, default=30,   help="最大的音频长度，单位秒")
add_arg("local_files_only",  type=bool,  default=True, help="是否只在本地加载模型，不尝试下载")
add_arg("noise",  type=bool,  default=False, help="输入模型的是噪声")
add_arg("task",       type=str, default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
# add_arg("metric",     type=str, default="cer",        choices=['cer', 'wer'],              help="评估方式")
args = parser.parse_args()
print_arguments(args)

# 获取Whisper的数据处理器，这个包含了特征提取器、tokenizer
processor = WhisperProcessor.from_pretrained(args.base_model,
                                             language=args.language,
                                             task=args.task,
                                             no_timestamps=not args.timestamps,
                                             local_files_only=args.local_files_only)
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language=args.language,
    task=args.task,
    no_timestamps=not args.timestamps,)
# 获取模型
model = WhisperForConditionalGeneration.from_pretrained(args.base_model,
                                                        device_map="auto",
                                                        local_files_only=args.local_files_only)
if "checkpoint" in args.lora_model:
    # 这就是放的完整路径
    outputdir=args.lora_model
    print('1')
else:
    outputdir = os.path.join(args.lora_model, os.path.basename(args.base_model))
    print('2')
if args.load_lora_model:
    if args.lora_model is not None:
        if "checkpoint" not in args.lora_model:
            # 搜索最新ckp
            print(f'outputdir:{outputdir}')
            resume_from_checkpoints = os.listdir(outputdir)
            print(f'resume_from_checkpoints:{resume_from_checkpoints}')
            if len(resume_from_checkpoints)>0:
                resume_from_checkpoint = max(resume_from_checkpoints, key=lambda x: int(os.path.basename(x).split('-')[-1]))
                print(f'resume_from_checkpoint:{resume_from_checkpoint}')
                resume_from_checkpoint=os.path.join(outputdir,resume_from_checkpoint)
                print(f'resume_from_checkpoint:{resume_from_checkpoint}')
        else:
            resume_from_checkpoint=outputdir
            model = PeftModel.from_pretrained(model, resume_from_checkpoint, local_files_only=args.local_files_only)
            model = model.merge_and_unload()
            print(f"已加载lora model:{resume_from_checkpoint}")
    else:
        print(f"路径:{outputdir}下无训练好的模型，已使用原模型")
print(model)
model.eval()

# model.model.encoder.post_init()
# model.model.encoder.init_weights()
# model.post_init()
# 获取测试数据
test_dataset = SpeechDataset(data_list_path=args.test_data,
                             processor=processor,
                             timestamps=args.timestamps,
                             # sample_rate=args.sampling_rate,
                             language=args.language,
                             min_duration=args.min_audio_len,
                             max_duration=args.max_audio_len)
print(f"测试数据：{len(test_dataset)}")
# 数据padding器
data_collator = DataCollatorOnlySpeechSeq2SeqWithPadding(processor=processor)
eval_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, collate_fn=data_collator,shuffle=True)
# eval_dataloader = eval_dataloader[:int(0.1*len(eval_dataloader))]

# 获取评估方法
metrics = []
# metric_files = ['bert_score','bleu','mer', 'my_rouge','perplexity', 'wer','word_info_lost','word_info_preserved']
metric_files = ['bleu','mer', 'my_rouge','wer','word_info_lost','word_info_preserved','en_cer']
# Load metrics
for metric_file in metric_files:
    metric = evaluate.load(f'metrics/{metric_file}.py')
    metrics.append(metric)

result_basename=f'results{"_base"if not args.load_lora_model else "_lora"}{"_noise"if args.noise else ""}'
# 开始评估
output_file=os.path.join(outputdir,f'{result_basename}.txt')
with open(output_file, "w") as f:
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                input_features = batch["input_features"].cuda()
                if args.noise:
                    input_features=torch.randn_like(input_features)

                generated_tokens = (
                    model.generate(
                        input_features=input_features,
                        do_sample=False,
                        # top_p=0.95,
                        repetition_penalty=5.0,
                        # condition_on_previous_text=0,
                        decoder_input_ids=batch["labels"][:, :4].cuda(),
                        forced_decoder_ids=forced_decoder_ids,
                        # prompt_ids=prompt_ids,
                        # bad_words_ids=bad_words_ids,
                        max_new_tokens=255).cpu().numpy())

                labels = batch["labels"].cpu().numpy()
                # print(f'labels:{labels}')
                labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
                # 将预测和实际的 token 转换为文本
                decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
                print('decoded_preds')
                print(decoded_preds)
                print('decoded_labels')
                print(decoded_labels)
                print('end')
                decoded_preds=remove_punctuation(decoded_preds)
                decoded_labels=remove_punctuation(decoded_labels)
                for pred, label in zip(decoded_preds, decoded_labels):
                    f.write(f"start********************************\n")
                    f.write(f"Predicted: {pred}\n")
                    f.write(f"True: {label}\n")
                    f.write(f"end==================================\n\n")
                for metric in metrics:
                    metric.add_batch(predictions=decoded_preds, references=decoded_labels)
# 计算评估结果
results={}
for metric in metrics:
    result = metric.compute()
    for key in result.keys():
        if type(result[key])==torch.Tensor:
            result[key]=result[key].item()
        results[key]=result[key]
print(f"评估结果：{results}")
json_file_path=os.path.join(outputdir,f'{result_basename}.json')
with open(json_file_path,'w') as f:
    json.dump(results,f)