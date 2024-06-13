import numpy as np
from nltk.translate.meteor_score import single_meteor_score
import evaluate
import datasets


def compute_metrics(preds, labels):
    scores = [single_meteor_score(label.split(),pred.split()) for pred,label in zip(preds,labels)]
    scores=np.mean(scores)
    scores={'meteor':scores}
    return scores


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