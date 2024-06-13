import re
from typing import Any, List, Dict, Union
import torch


def list_operation(text: str or List[str],func):
    if isinstance(text, str):
        text = func(text)
        return text
    elif isinstance(text, list):
        result_text = []
        for t in text:
            t = func(t)
            result_text.append(t)
        return result_text
    else:
        raise Exception(f'不支持该类型{type(text)}')


def filter_ascii_str(text):
    return re.sub(r'[^a-zA-Z ]', '', text)


def filter_ascii_text(text: str or List[str]):
    if isinstance(text, str):
        text = filter_ascii_str(text)
        return text
    elif isinstance(text, list):
        result_text = []
        for t in text:
            t = filter_ascii_str(t)
            result_text.append(t)
        return result_text
    else:
        raise Exception(f'不支持该类型{type(text)}')


def convert_lower_text(text: str or List[str]):
    if isinstance(text, str):
        return text.lower()
    elif isinstance(text, list):
        result_text = []
        for t in text:
            result_text.append(t.lower())
        return result_text
    else:
        raise Exception(f'不支持该类型{type(text)}')

def model_generate(model,batch,strategy,args,**kwargs):

    input_features = batch["input_features"].cuda()
    if args.noise:
        input_features = torch.randn_like(input_features)
    if strategy=='greedy':
        output=model.generate(input_features,do_sample=False,num_beams=1,**kwargs)
    elif strategy=='beamSearch':
        output=model.generate(input_features,do_sample=False,num_beams=20,**kwargs)
    elif strategy=='multinomialSampling':
        output=model.generate(input_features,do_sample=True,num_beams=20,**kwargs)
    elif strategy=='topkSampling':
        output=model.generate(input_features,do_sample=True,num_beams=20,top_k=5,**kwargs)
    elif strategy=='toppSampling':
        output=model.generate(input_features,do_sample=True,num_beams=20,top_p=0.5,**kwargs)
    elif strategy=='contrastiveSearch':
        output=model.generate(input_features,penalty_alpha=1,top_k=5,**kwargs)
    else:
        raise NotImplementedError
    return output
