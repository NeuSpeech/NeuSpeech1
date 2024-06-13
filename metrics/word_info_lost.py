from torchmetrics.text import WordInfoLost

import evaluate
import datasets
wer  = WordInfoLost()




def compute_metrics(preds, labels):


    scores = wer(preds, labels)
    scores={'wil':scores.item()}
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