import torch
from transformers.trainer_pt_utils import LabelSmoother
import torch.nn as nn
import torch.nn.functional as F
import os

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def projection_module(config_name='',**kwargs):
    if config_name=='base':
        d_model = kwargs['d_model']
        conv1 = nn.Sequential(
            nn.Conv1d(kwargs['meg_ch'], d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
        )
        conv1.stride = (2,)
    elif config_name=='replace':
        d_model = kwargs['d_model']
        conv1 = nn.Conv1d(kwargs['meg_ch'], d_model, kernel_size=3, stride=2, padding=1)
    else:
        raise NotImplementedError
    return conv1

def load_from_checkpoint(resume_from_checkpoint, model=None):
    pass


def save_model(self, output_dir = None, _internal_call: bool = False):
    pass


def trainer_save_model(output_dir=None, state_dict=None):
    os.makedirs(output_dir, exist_ok=True)


def compute_accuracy(pred):
    ## 1.处理 pred.predictions
    # 每个样本的预测结果为vocab大小
    predict_res = torch.Tensor(pred.predictions[0])  # size：[验证集样本量, label的token长度, vocab大小]
    pred_ids = predict_res.argmax(dim=2)

    ## 2.处理 pred.label_ids
    labels_actual = torch.LongTensor(pred.label_ids)

    ## 3.计算accuracy
    total_num = labels_actual.shape[0]
    acc = torch.sum(torch.all(torch.eq(pred_ids, labels_actual), dim=1)) / total_num
    return {'accuracy': acc}

