import jsonlines
import os
import sys
import numpy as np
# 获取当前脚本的文件路径
current_path = os.path.abspath(__file__)
# 获取项目根目录的路径
project_root = os.path.dirname(os.path.dirname(current_path))
# 将项目根目录添加到 sys.path
sys.path.append(project_root)

import argparse
import functools
from utils.utils import add_arguments


def write_jsonlines(file_path, json_dicts):
    with jsonlines.open(file_path, mode='w') as writer:
        for json_dict in json_dicts:
            writer.write(json_dict)


def read_jsonlines(file_path):
    json_dicts = []
    with jsonlines.open(file_path, mode='r') as reader:
        for json_obj in reader:
            json_dicts.append(json_obj)
    return json_dicts

def makedirs(path):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    return path

if __name__ == '__main__':
    home_dir = os.path.expanduser("~")
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg("jsonl",    type=str, nargs="+", default=[],       help="jsonl文件路径")
    add_arg("output_jsonl",     type=str, default=None,        help="存储jsonl文件路径")
    add_arg("shuffle",     type=bool, default=True,        help="打乱顺序")
    args = parser.parse_args()
    output_jsonl=[]
    for json in args.jsonl:
        json=os.path.join(home_dir,json)
        datas = read_jsonlines(json)
        output_jsonl.extend(datas)
    if args.shuffle:
        np.random.shuffle(output_jsonl)
    write_jsonlines(makedirs(os.path.join(home_dir,args.output_jsonl)), output_jsonl)

