import lmppl

import datasets
import numpy as np
import evaluate

scorer = lmppl.LM('gpt2')


def compute_metrics(preds, labels):
    scores_p = scorer.get_perplexity(preds)
    scores_l = scorer.get_perplexity(labels)
    scores={'ppl_preds':np.mean(scores_p),'ppl_labels':np.mean(scores_l)}
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