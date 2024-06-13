from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from metrics.chinese_bert_score import score as bert_score_f
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import numpy as np
import nltk
import evaluate
import datasets
# nltk.download('wordnet2021')
# nltk.download('omw-1.4')
from nltk.corpus import wordnet2021 as wn

from typing import List
import jiwer
import jiwer.transforms as tr
from datasets.config import PY_VERSION
from packaging import version

import evaluate


if PY_VERSION < version.parse("3.8"):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


SENTENCE_DELIMITER = ""


if version.parse(importlib_metadata.version("jiwer")) < version.parse("2.3.0"):

    class SentencesToListOfCharacters(tr.AbstractTransform):
        def __init__(self, sentence_delimiter: str = " "):
            self.sentence_delimiter = sentence_delimiter

        def process_string(self, s: str):
            return list(s)

        def process_list(self, inp: List[str]):
            chars = []
            for sent_idx, sentence in enumerate(inp):
                chars.extend(self.process_string(sentence))
                if self.sentence_delimiter is not None and self.sentence_delimiter != "" and sent_idx < len(inp) - 1:
                    chars.append(self.sentence_delimiter)
            return chars

    cer_transform = tr.Compose(
        [tr.RemoveMultipleSpaces(), tr.Strip(), SentencesToListOfCharacters(SENTENCE_DELIMITER)]
    )
else:
    cer_transform = tr.Compose(
        [
            tr.RemoveMultipleSpaces(),
            tr.Strip(),
            tr.ReduceToSingleSentence(SENTENCE_DELIMITER),
            tr.ReduceToListOfListOfChars(),
        ]
    )




def compute_metrics(preds, labels):
    # 字符级别
    decoded_preds = [" ".join((pred.replace(" ", ""))) for pred in preds]
    decoded_labels = [" ".join((label.replace(" ", ""))) for label in labels]
    # 词级别，分词
    # decoded_preds = [" ".join(jieba.cut(pred.replace(" ", ""))) for pred in decoded_preds]
    # decoded_labels = [" ".join(jieba.cut(label.replace(" ", ""))) for label in decoded_labels]
    rouge = Rouge()
    labels_lens = [len(pred) for pred in labels]

    total = 0

    cer_incorrect = 0
    cer_total = 0
    wer_incorrect = 0
    wer_total = 0
    rouge_1, rouge_2, rouge_l, bleu, meteor_sum = 0, 0, 0, 0, 0
    for decoded_label, decoded_pred in zip(decoded_labels, decoded_preds):
        total += 1

        measures = jiwer.compute_measures(
            decoded_label,
            decoded_pred,
            truth_transform=cer_transform,
            hypothesis_transform=cer_transform,
        )
        cer_incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
        cer_total += measures["substitutions"] + measures["deletions"] + measures["hits"]
        measures = jiwer.compute_measures(decoded_label, decoded_pred)
        wer_incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
        wer_total += measures["substitutions"] + measures["deletions"] + measures["hits"]
        scores = rouge.get_scores(hyps=decoded_pred, refs=decoded_label)
        rouge_1 += scores[0]['rouge-1']['f']
        rouge_2 += scores[0]['rouge-2']['f']
        rouge_l += scores[0]['rouge-l']['f']
        bleu += sentence_bleu(
            references=[decoded_label.split(' ')],
            hypothesis=decoded_pred.split(' '),
            smoothing_function=SmoothingFunction().method1
        )
        meteor_sum += meteor_score([decoded_label.split()], decoded_pred.split(),wordnet=wn)

    bleu /= total
    rouge_1 /= total
    rouge_2 /= total
    rouge_l /= total
    meteor_sum /= total
    P, R, F1 = bert_score_f(decoded_preds, decoded_labels, model_path='./metrics/bert-base-chinese', lang="zh",
                            verbose=False)
    result = {'rouge-1': rouge_1, 'rouge-2': rouge_2, 'rouge-l': rouge_l, 'bert-score': F1.mean().item(), 'bleu': bleu,
              'meteor_sum': meteor_sum, "gen_len": np.mean(labels_lens), "cer":cer_incorrect / cer_total,'wer':wer_incorrect/wer_total}
    return result


class FullEval(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description='None',
            citation='None',
            inputs_description='None',
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=[""],
            reference_urls=[
                "",
            ],
        )
    def _compute(self, predictions, references):

        return compute_metrics(predictions,references)