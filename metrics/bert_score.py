from bert_score import score
import datasets

import evaluate



def compute_metrics(preds, labels):
    scores = score(preds, labels,lang='en')
    scores={
        "bert_p":scores[0].mean()*100,
        "bert_r":scores[1].mean()*100,
        "bert_f":scores[2].mean()*100,
    }
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