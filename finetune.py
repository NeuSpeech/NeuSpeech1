import argparse
import functools
import os
import platform
import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
# import torch._dynamo as dynamo
import logging
from peft import LoraConfig, get_peft_model, AdaLoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperProcessor
from utils.callback import SavePeftModelCallback
from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding,get_part_of_dataset
from utils.model_utils import load_from_checkpoint,trainer_save_model,compute_accuracy,projection_module
from utils.load_model import WhisperForConditionalGeneration,match_modules,match_modules_string
from utils.reader import CustomDataset
from utils.utils import print_arguments, make_inputs_require_grad, add_arguments


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("train_data",    type=str, default="/home/yyang/dataset/multi_media/formal_dataset/cut/train_data.jsonl",       help="训练数据集的路径")
add_arg("test_data",     type=str, default="/home/yyang/dataset/multi_media/formal_dataset/cut/val_data.jsonl",        help="测试数据集的路径")
add_arg("base_model",    type=str, default="/home/yyang/dataset/multi_media/transformers_whisper_models/large_finetune",      help="Whisper的基础模型")
add_arg("lora_model",    type=str, default=None,      help="训练过的lora模型")
add_arg("output_dir",    type=str, default="output1/",                  help="训练保存模型的路径")
add_arg("warmup_steps",  type=int, default=10000,      help="训练预热步数")
add_arg("logging_steps", type=int, default=100,     help="打印日志步数")
add_arg("eval_steps",    type=int, default=1000,    help="多少步数评估一次")
add_arg("save_steps",    type=int, default=1000,    help="多少步数保存模型一次")
add_arg("num_workers",   type=int, default=6,       help="读取数据的线程数量")
add_arg("learning_rate", type=float, default=1e-3,  help="学习率大小")
add_arg("modal", type=str, default='speech',  help="输入的模态")
add_arg("sampling_rate", type=int, default=200,  help="输入信号期望采样率")
add_arg("orig_sample_rate", type=int, default=200,  help="输入信号采样率")
add_arg("eeg_ch", type=int, default=224,  help="输入信号通道数")
add_arg("lora_eeg_ch", type=int, default=None,  help="lora模型的输入信号通道数")
add_arg("min_audio_len", type=float, default=0.5,   help="最小的音频长度，单位秒")
add_arg("max_audio_len", type=float, default=30,    help="最大的音频长度，单位秒")
add_arg("use_adalora",   type=bool,  default=True,  help="是否使用AdaLora而不是Lora")
add_arg("fp16",          type=bool,  default=False,  help="是否使用fp16训练模型")
add_arg("use_8bit",      type=bool,  default=False, help="是否将模型量化为8位")
add_arg("filter_dataset",      type=bool,  default=False, help="是否过滤数据集")
add_arg("timestamps",    type=bool,  default=True, help="训练时是否使用时间戳数据")
add_arg("local_files_only", type=bool, default=True, help="是否只在本地加载模型，不尝试下载")
add_arg("num_train_epochs", type=int, default=30,      help="训练的轮数")
add_arg("language",      type=str, default="English", help="设置语言，可全称也可简写，如果为None则训练的是多语言")
add_arg("task",     type=str, default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
add_arg("augment_config_path",         type=str, default='configs/augmentation.json', help="数据增强配置文件路径")
add_arg("resume_from_checkpoint",      type=str, default=None, help="恢复训练的检查点路径")
add_arg("per_device_train_batch_size", type=int, default=2,    help="训练的batch size")
add_arg("per_device_eval_batch_size",  type=int, default=2,    help="评估的batch size")
add_arg("gradient_accumulation_steps", type=int, default=1,    help="梯度累积步数")
add_arg("fine_tune_layers", type=int, default=None,    help="微调base model的前多少层")
add_arg("device", type=str, default='auto',    help="device")
add_arg("config_name", type=str, default='base',    help="conv1 module")
add_arg("data_ratio", type=float, default=None,    help="训练集使用数据的比例")
add_arg("random_initialize_whisper", type=bool, default=False,    help="随机初始化whisper")
add_arg("combine_sentences", type=bool, default=False,    help="训练时增加多个句子")
add_arg("split_sentences", type=bool, default=False,    help="训练时将句子拆分")
add_arg("ft_full", type=bool, default=False,    help="微调整个模型")
args = parser.parse_args()
print_arguments(args)


# 获取Whisper的数据处理器，这个包含了特征提取器、tokenizer
processor = WhisperProcessor.from_pretrained(args.base_model,
                                             language=args.language,
                                             task=args.task,
                                             no_timestamps=not args.timestamps,
                                             local_files_only=args.local_files_only)

# 读取数据
train_dataset = CustomDataset(
    data_list_path=args.train_data,
    processor=processor,
    modal=args.modal,
    modal_ch=args.eeg_ch,
    mode='train',
    sample_rate=args.sampling_rate,
    orig_sample_rate=args.orig_sample_rate,
    language=args.language,
    filter_dataset=args.filter_dataset,
    timestamps=args.timestamps,
    combine_sentences=args.combine_sentences,
    split_sentences=args.split_sentences,
    min_duration=args.min_audio_len,
    max_duration=args.max_audio_len,
    augment_config_path=args.augment_config_path)
test_dataset = CustomDataset(
    data_list_path=args.test_data,
    processor=processor,
    modal=args.modal,
    mode='val',
    modal_ch=args.eeg_ch,
    sample_rate=args.sampling_rate,
    orig_sample_rate=args.orig_sample_rate,
    language=args.language,
    filter_dataset=args.filter_dataset,
    timestamps=args.timestamps,
    min_duration=args.min_audio_len,
    max_duration=args.max_audio_len)

if args.data_ratio is not None:
    train_dataset.data_list=get_part_of_dataset(train_dataset.data_list,args.data_ratio)

print(f"训练数据：{len(train_dataset)}，测试数据：{len(test_dataset)}")
# 数据padding器
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 获取Whisper模型
device_map = args.device
if device_map == 'cpu':
    ddp=0
else:
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
# print(f'device_map:{device_map}, os env:{os.environ["CUDA_VISIBLE_DEVICES"]}')
# device_map = 'cpu'
# 获取模型
print(f'device map :{device_map}')
model=WhisperForConditionalGeneration.from_pretrained(args.base_model,
                                                    load_in_8bit=args.use_8bit,
                                                    device_map=device_map,
                                                    local_files_only=args.local_files_only,
                                                        )
print(f'model device {model.device}')
eeg_ch=args.eeg_ch
if args.lora_eeg_ch is not None:
    eeg_ch=args.lora_eeg_ch

device=model.device
kwargs={
    'meg_ch':eeg_ch,
    'd_model':model.model.encoder.conv2.in_channels,
}

conv1=projection_module(config_name=args.config_name,**kwargs)


# conv1 = nn.Conv1d(meg_ch, d_model, kernel_size=3, padding=1)
conv1 = conv1.to(device)
model.model.encoder.set_input_embeddings(conv1)
# model=model.to(device)
if args.lora_model is not None:
    # 之前的加载模型是把模型变成要加载的模型的形状，然后再加载参数。
    # 现在是变成要训练的模型。
    model = PeftModel.from_pretrained(model, args.lora_model, local_files_only=args.local_files_only)
    model = model.merge_and_unload()
    if args.lora_eeg_ch!=args.eeg_ch:
        kwargs={
            'meg_ch':args.eeg_ch,
            'd_model':model.model.encoder.conv2.in_channels,
        }

        conv1=projection_module(config_name=args.config_name,**kwargs)
        conv1 = conv1.to(device)
        model.model.encoder.set_input_embeddings(conv1)
if args.random_initialize_whisper:
    model.post_init() #todo 这个没用，还不会弄
    print('模型已被初始化')
# model.save_pretrained(save_directory=os.path.join(args.output_dir, "checkpoint-init"))
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
# 量化模型
model = prepare_model_for_kbit_training(model)
# 注册forward，否则多卡训练会失败
model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
# model.model.encoder.conv1[0].register_forward_hook(make_inputs_require_grad)

for param in model.parameters():
    param.requires_grad=False
# for param in model.model.model.decoder.parameters():
#     param.requires_grad=False

print('加载LoRA模块...')
if args.resume_from_checkpoint:
    # 恢复训练时加载Lora参数
    print("Loading adapters from checkpoint.")
    model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
else:
    print(f'adding LoRA modules...')
    # prefixes = [f'model.encoder.layers.{i}.' for i in [0,1,2,3]]
    if args.fine_tune_layers is not None:
        prefixes = [f'model.encoder.layers.{i}.' for i in range(args.fine_tune_layers)]
    elif args.ft_full:
        prefixes = ['model']
    else:
        prefixes = ['model.encoder']
    suffixes = ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]
    # model_named_modules=[]
    # target_modules = []
    target_modules = match_modules_string(model.named_modules(), prefixes, suffixes)
    print('target_modules')
    print(target_modules)
    # modules_to_save= match_modules(model.named_modules(),[''],[''],[".*model.encoder.conv1",".*model.encoder.conv2"])
    modules_to_save= ['model.encoder.conv1', 'model.encoder.conv2']
    print('modules_to_save')
    print(modules_to_save)
    if args.use_adalora:
        config = AdaLoraConfig(init_r=12, target_r=4, beta1=0.85, beta2=0.85, tinit=200, tfinal=1000, deltaT=10,
                               lora_alpha=32, lora_dropout=0.1, orth_reg_weight=0.5, target_modules=target_modules,
                               modules_to_save=modules_to_save)
    else:
        config = LoraConfig(r=32, lora_alpha=64, target_modules=target_modules, lora_dropout=0.05, bias="none",
                            modules_to_save=modules_to_save)
    model = get_peft_model(model, config)
# model.to('cpu')
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total trainable parameters: {trainable_params}')
# 手动关闭一下original_module的梯度
params_to_no_grad = match_modules(model.named_parameters(), [''],
                                    [''],['original_module'])

for name,param in model.named_parameters():
    if name in params_to_no_grad:
        param.requires_grad=False
print('*'*100)
print(model.print_trainable_parameters())
print('^'*100)

if args.base_model.endswith("/"):
    args.base_model = args.base_model[:-1]
output_dir = os.path.join(args.output_dir, os.path.basename(args.base_model))
# 定义训练参数
training_args = \
    Seq2SeqTrainingArguments(output_dir=output_dir,  # 保存检查点和意志的目录
                             per_device_train_batch_size=args.per_device_train_batch_size,  # 训练batch_size大小
                             per_device_eval_batch_size=args.per_device_eval_batch_size,  # 评估batch_size大小
                             gradient_accumulation_steps=args.gradient_accumulation_steps,  # 训练梯度累计步数
                             learning_rate=args.learning_rate,  # 学习率大小
                             warmup_steps=args.warmup_steps,  # 预热步数
                             num_train_epochs=args.num_train_epochs,  # 微调训练轮数
                             save_strategy="steps",  # 指定按照步数保存检查点
                             evaluation_strategy="steps",  # 指定按照步数评估模型
                             load_best_model_at_end=False,  # 指定是否在结束时加载最优模型
                             fp16=args.fp16,  # 是否使用半精度训练
                             report_to=["tensorboard"],  # 指定使用tensorboard保存log
                             save_steps=args.save_steps,  # 指定保存检查点的步数
                             eval_steps=args.eval_steps,  # 指定评估模型的步数
                             save_total_limit=5,  # 只保存最新检查点的数量
                             optim='adamw_torch',  # 指定优化方法
                             ddp_find_unused_parameters=False if ddp else None,  # 分布式训练设置
                             dataloader_num_workers=args.num_workers,  # 设置读取数据的线程数量
                             logging_steps=args.logging_steps,  # 指定打印log的步数
                             remove_unused_columns=False,  # 删除模型不需要的数据列
                             label_names=["labels"],
                             )  # 与标签对应的输入字典中的键列表

# if training_args.local_rank == 0 or training_args.local_rank == -1:
print('trainable parameters')
print('=' * 90)

for name,param in model.named_parameters():
    if param.requires_grad:
        print(name)
print('=' * 90)

# 使用Pytorch2.0的编译器
# if torch.__version__ >= "2" and platform.system().lower() != 'windows':
#     model = torch.compile(model)

# 定义训练器
trainer = Seq2SeqTrainer(args=training_args,
                         model=model,
                         train_dataset=train_dataset,
                         eval_dataset=test_dataset,
                         data_collator=data_collator,
                         tokenizer=processor.feature_extractor,
                         callbacks=[SavePeftModelCallback],
                         # compute_metrics=compute_metrics
                         )
model.config.use_cache = False
trainer._load_from_checkpoint = load_from_checkpoint
# 因为有新的权重，我们需要从开始保存。
# model.save_pretrained(os.path.join(output_dir, "checkpoint-init"))
# 开始训练
# trainer.save_model(output_dir=os.path.join(output_dir, "checkpoint-init"))
# trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
# 训练荷兰语总是因为不知名的原因被终止。我现在要让他有10次接着跑的机会
resume_from_checkpoint=args.resume_from_checkpoint
# resume_from_checkpoint_dir=os.path.dirname(resume_from_checkpoint)
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
# if args.device=='cpu':
#     trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
# else:
#     for i in range(10):
#         try:
#             # 搜索最新的cp
#             # 如果输出的模型已经保存了，就把那里当做resume_from_checkpoint
#             if len(os.listdir(output_dir))>0:
#                 resume_from_checkpoint=output_dir
#             resume_from_checkpoints=os.listdir(resume_from_checkpoint)
#             resume_from_checkpoint = max(resume_from_checkpoints, key=lambda x: int(os.path.basename(x).split('-')[-1]))
#             resume_from_checkpoint=os.path.join(resume_from_checkpoint_dir,resume_from_checkpoint)
#             # resume_from_checkpoints_number=[eval(os.path.basename(x).split('-')[-1]) for x in resume_from_checkpoints]
#             trainer.train(resume_from_checkpoint=resume_from_checkpoint)
#         except Exception as e:
#             print(e)
trainer.save_model(os.path.join(output_dir, "checkpoint-final"))
# 保存最后的模型
# trainer.save_state()
# if training_args.local_rank == 0 or training_args.local_rank == -1:
#     model.save_pretrained(os.path.join(output_dir, "checkpoint-final"))
