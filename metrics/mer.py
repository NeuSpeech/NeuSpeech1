from torchmetrics.text import MatchErrorRate
import evaluate
import datasets

wer = MatchErrorRate()


def compute_metrics(preds, labels):


    scores = wer(preds, labels)
    scores={'mer':scores.item()}
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