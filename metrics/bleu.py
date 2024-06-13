from torchmetrics.text import BLEUScore
import datasets
import evaluate


def compute_metrics(preds, labels):
    labels=[[label] for i,label in enumerate(labels)]
    result={}
    for i in range(1,5):
        bleu=BLEUScore(n_gram=i)
        result[f'bleu-{i}']=bleu(preds,labels)
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