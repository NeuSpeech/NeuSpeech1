from torchmetrics.functional.text.rouge import rouge_score
import datasets
import evaluate


def compute_metrics(preds, labels):

    metrics={}
    for decoded_label, decoded_pred in zip(labels, preds):
        metric=rouge_score(decoded_pred,decoded_label)
        for key in metric.keys():
            metrics[key]=metrics.get(key, 0) + metric[key]
    for key in metrics.keys():
        metrics[key]=metrics[key]/len(labels)*100

    return metrics


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