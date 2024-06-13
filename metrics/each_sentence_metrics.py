
import evaluate
import os
import sys
import torch
# 获取当前脚本的文件路径
current_path = os.path.abspath(__file__)
# 获取项目根目录的路径
project_root = os.path.dirname(os.path.dirname(current_path))
# 将项目根目录添加到 sys.path
sys.path.append(project_root)
from utils.data_utils import generate_random_string


class EachSentenceMetrics:
    def __init__(self,metrics_files=('bleu','mer', 'my_rouge','wer','word_info_lost','word_info_preserved')):
        self.metrics_files=metrics_files
        self.metrics=self.load_metrics()

    def load_metrics(self):
        metrics = []
        metric_files = self.metrics_files
        # Load metrics
        for metric_file in metric_files:
            metric = evaluate.load(f'metrics/{metric_file}.py',
                                   experiment_id=generate_random_string(100))
            metrics.append(metric)
        return metrics

    def compute(self,predictions,references):
        all_sentences_metrics_list=[]
        for i,(p,r) in enumerate(zip(predictions,references)):
            results = {}
            for metric in self.metrics:
                result = metric.compute(predictions=[p],references=[r])
                for key in result.keys():
                    if type(result[key]) == torch.Tensor:
                        result[key] = result[key].item()
                    results[key] = result[key]
            all_sentences_metrics_list.append(results)
        return all_sentences_metrics_list



