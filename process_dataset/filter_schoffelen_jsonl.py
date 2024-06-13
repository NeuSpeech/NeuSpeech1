import jsonlines
import os
import sys
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
    add_arg("output_dir",    type=str,  default=None,       help="输出jsonl文件夹")
    args = parser.parse_args()
    for json in args.jsonl:
        datas = read_jsonlines(os.path.join(home_dir,json))
        datas=[data for data in datas if data['sent_type']=="ZINNEN"]
        if args.output_dir is not None:
            json=os.path.join(home_dir,args.output_dir,os.path.basename(json))
        write_jsonlines(makedirs(json), datas)

