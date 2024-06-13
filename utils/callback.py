import os
import shutil

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from utils.reader import write_jsonlines
import torch


# 保存模型时的回调函数
class SavePeftModelCallback(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # 首先要检查是不是要保存的时候
        control.should_save=False
        if state.global_step % state.save_steps == 0:
            log_history = []
            for i, data in enumerate(state.log_history):
                if 'eval_loss' in data.keys():
                    log_history.append(data['eval_loss'])
            if len(log_history)>0:  # 因为训练久了，best_metric会变null，所以得自己弄
                if log_history[-1]==min(log_history):
                    control.should_save=True

                # if os.path.exists(state.best_model_checkpoint):
                #     shutil.rmtree(state.best_model_checkpoint)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # epoch结束不准存
        control.should_save=False

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_save=False

    def on_save(self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                **kwargs, ):
            # 复制Lora模型，主要是兼容旧版本的peft
        # args.should_save=False
        # checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        # if os.path.exists(checkpoint_folder):
        #     shutil.rmtree(checkpoint_folder)
        # if len(state.log_history)>0:
        #     now_metric=state.log_history[-1]['eval_loss']
        # else:
        #     now_metric=None
        # if now_metric==state.best_metric:
        #     # 这是最好的
        #     kwargs["model"].save_pretrained(checkpoint_folder)
        #     write_jsonlines(f'{checkpoint_folder}/history.txt',state.log_history)
        return control


class SaveFullModelCallback(TrainerCallback):
    def on_save(self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                **kwargs, ):
        if args.local_rank == 0 or args.local_rank == -1:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            kwargs["model"].save_pretrained(checkpoint_folder)
            # 保存效果最好的模型
            best_checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-best")
            # 因为只保存最新5个检查点，所以要确保不是之前的检查点
            if os.path.exists(state.best_model_checkpoint):
                if os.path.exists(best_checkpoint_folder):
                    shutil.rmtree(best_checkpoint_folder)
                shutil.copytree(state.best_model_checkpoint, best_checkpoint_folder)
            print(f"效果最好的检查点为：{state.best_model_checkpoint}，评估结果为：{state.best_metric}")
        return control
