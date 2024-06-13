import os
import pathlib
import sys
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from transformers import AutoTokenizer

from transformers import (AutoModel, AutoTokenizer, BertConfig, GPT2Tokenizer, RobertaTokenizer,
                          RobertaConfig, XLMConfig, XLNetConfig)
from bert_score.utils import (cache_scibert, get_bert_embedding,
                              get_hash, get_idf_dict, lang2model, model2layers,
                              sent_encode,tqdm,pad_sequence,greedy_cos_idf)

from packaging import version
from transformers import __version__ as trans_version

__all__ = ["score", "plot_example"]


def bert_cos_score_idf(
    model,
    refs,
    hyps,
    tokenizer,
    idf_dict,
    verbose=False,
    batch_size=64,
    device="cuda:0",
    all_layers=False,
):
    """
    Compute BERTScore.

    Args:
        - :param: `model` : a BERT model in `pytorch_pretrained_bert`
        - :param: `refs` (list of str): reference sentences
        - :param: `hyps` (list of str): candidate sentences
        - :param: `tokenzier` : a BERT tokenizer corresponds to `model`
        - :param: `idf_dict` : a dictionary mapping a word piece index to its
                               inverse document frequency
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """
    preds = []

    def dedup_and_sort(l):
        return sorted(list(set(l)), key=lambda x: len(x.split(" ")), reverse=True)

    sentences = dedup_and_sort(refs + hyps)
    embs = []
    iter_range = range(0, len(sentences), batch_size)
    if verbose:
        print("computing bert embedding.")
        iter_range = tqdm(iter_range)
    stats_dict = dict()
    for batch_start in iter_range:
        sen_batch = sentences[batch_start : batch_start + batch_size]
        embs, masks, padded_idf = get_bert_embedding(
            sen_batch, model, tokenizer, idf_dict, device=device, all_layers=all_layers
        )
        embs = embs.cpu()
        masks = masks.cpu()
        padded_idf = padded_idf.cpu()
        for i, sen in enumerate(sen_batch):
            sequence_len = masks[i].sum().item()
            emb = embs[i, :sequence_len]
            idf = padded_idf[i, :sequence_len]
            stats_dict[sen] = (emb, idf)

    def pad_batch_stats(sen_batch, stats_dict, device):
        stats = [stats_dict[s] for s in sen_batch]
        emb, idf = zip(*stats)
        emb = [e.to(device) for e in emb]
        idf = [i.to(device) for i in idf]
        lens = [e.size(0) for e in emb]
        emb_pad = pad_sequence(emb, batch_first=True, padding_value=2.0)
        idf_pad = pad_sequence(idf, batch_first=True)

        def length_to_mask(lens):
            lens = torch.tensor(lens, dtype=torch.long)
            max_len = max(lens)
            base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len)
            return base < lens.unsqueeze(1)

        pad_mask = length_to_mask(lens).to(device)
        return emb_pad, pad_mask, idf_pad

    device = next(model.parameters()).device
    iter_range = range(0, len(refs), batch_size)
    if verbose:
        print("computing greedy matching.")
        iter_range = tqdm(iter_range)

    with torch.no_grad():
        for batch_start in iter_range:
            batch_refs = refs[batch_start : batch_start + batch_size]
            batch_hyps = hyps[batch_start : batch_start + batch_size]
            ref_stats = pad_batch_stats(batch_refs, stats_dict, device)
            hyp_stats = pad_batch_stats(batch_hyps, stats_dict, device)
            P, R, F1 = greedy_cos_idf(*ref_stats, *hyp_stats, all_layers)
            preds.append(torch.stack((P, R, F1), dim=-1).cpu())
    preds = torch.cat(preds, dim=1 if all_layers else 0)
    return preds


def get_tokenizer(model_type, use_fast=False):
    if model_type.startswith("scibert"):
        model_type = cache_scibert(model_type)

    if version.parse(trans_version) >= version.parse("4.0.0"):
        tokenizer = AutoTokenizer.from_pretrained(model_type, use_fast=use_fast)
    else:
        assert not use_fast, "Fast tokenizer is not available for version < 4.0.0"
        tokenizer = AutoTokenizer.from_pretrained(model_type)

    return tokenizer


def get_model(model_type, num_layers, all_layers=None):
    if model_type.startswith("scibert"):
        model = AutoModel.from_pretrained(cache_scibert(model_type))
    elif "t5" in model_type:
        from transformers import T5EncoderModel

        model = T5EncoderModel.from_pretrained(model_type)
    else:
        model = AutoModel.from_pretrained(model_type)
    model.eval()

    if hasattr(model, "decoder") and hasattr(model, "encoder"):
        model = model.encoder

    # drop unused layers
    if not all_layers:
        if hasattr(model, "n_layers"):  # xlm
            assert (
                0 <= num_layers <= model.n_layers
            ), f"Invalid num_layers: num_layers should be between 0 and {model.n_layers} for {model_type}"
            model.n_layers = num_layers
        elif hasattr(model, "layer"):  # xlnet
            assert (
                0 <= num_layers <= len(model.layer)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.layer)} for {model_type}"
            model.layer = torch.nn.ModuleList(
                [layer for layer in model.layer[:num_layers]]
            )
        elif hasattr(model, "encoder"):  # albert
            if hasattr(model.encoder, "albert_layer_groups"):
                assert (
                    0 <= num_layers <= model.encoder.config.num_hidden_layers
                ), f"Invalid num_layers: num_layers should be between 0 and {model.encoder.config.num_hidden_layers} for {model_type}"
                model.encoder.config.num_hidden_layers = num_layers
            elif hasattr(model.encoder, "block"):  # t5
                assert (
                    0 <= num_layers <= len(model.encoder.block)
                ), f"Invalid num_layers: num_layers should be between 0 and {len(model.encoder.block)} for {model_type}"
                model.encoder.block = torch.nn.ModuleList(
                    [layer for layer in model.encoder.block[:num_layers]]
                )
            else:  # bert, roberta
                assert (
                    0 <= num_layers <= len(model.encoder.layer)
                ), f"Invalid num_layers: num_layers should be between 0 and {len(model.encoder.layer)} for {model_type}"
                model.encoder.layer = torch.nn.ModuleList(
                    [layer for layer in model.encoder.layer[:num_layers]]
                )
        elif hasattr(model, "transformer"):  # bert, roberta
            assert (
                0 <= num_layers <= len(model.transformer.layer)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.transformer.layer)} for {model_type}"
            model.transformer.layer = torch.nn.ModuleList(
                [layer for layer in model.transformer.layer[:num_layers]]
            )
        elif hasattr(model, "layers"):  # bart
            assert (
                0 <= num_layers <= len(model.layers)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.layers)} for {model_type}"
            model.layers = torch.nn.ModuleList(
                [layer for layer in model.layers[:num_layers]]
            )
        else:
            raise ValueError("Not supported")
    else:
        if hasattr(model, "output_hidden_states"):
            model.output_hidden_states = True
        elif hasattr(model, "encoder"):
            model.encoder.output_hidden_states = True
        elif hasattr(model, "transformer"):
            model.transformer.output_hidden_states = True
        # else:
        #     raise ValueError(f"Not supported model architecture: {model_type}")

    return model


def score(
    cands,
    refs,
    model_path=None,
    num_layers=None,
    verbose=False,
    idf=False,
    device=None,
    batch_size=64,
    nthreads=4,
    all_layers=False,
    lang=None,
    return_hash=False,
    rescale_with_baseline=False,
    baseline_path=None,
    use_fast_tokenizer=False,
):
    """
    BERTScore metric.

    Args:
        - :param: `cands` (list of str): candidate sentences
        - :param: `refs` (list of str or list of list of str): reference sentences
        - :param: `model_type` (str): bert specification, default using the suggested
                  model for the target langauge; has to specify at least one of
                  `model_type` or `lang`
        - :param: `num_layers` (int): the layer of representation to use.
                  default using the number of layer tuned on WMT16 correlation data
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `idf` (bool or dict): use idf weighting, can also be a precomputed idf_dict
        - :param: `device` (str): on which the contextual embedding model will be allocated on.
                  If this argument is None, the model lives on cuda:0 if cuda is available.
        - :param: `nthreads` (int): number of threads
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `lang` (str): language of the sentences; has to specify
                  at least one of `model_type` or `lang`. `lang` needs to be
                  specified when `rescale_with_baseline` is True.
        - :param: `return_hash` (bool): return hash code of the setting
        - :param: `rescale_with_baseline` (bool): rescale bertscore with pre-computed baseline
        - :param: `baseline_path` (str): customized baseline file
        - :param: `use_fast_tokenizer` (bool): `use_fast` parameter passed to HF tokenizer

    Return:
        - :param: `(P, R, F)`: each is of shape (N); N = number of input
                  candidate reference pairs. if returning hashcode, the
                  output will be ((P, R, F), hashcode). If a candidate have
                  multiple references, the returned score of this candidate is
                  the *best* score among all references.
    """
    assert len(cands) == len(refs), "Different number of candidates and references"

    model_type=os.path.basename(model_path)
    assert (
        lang is not None or model_type is not None
    ), "Either lang or model_type should be specified"

    ref_group_boundaries = None
    if not isinstance(refs[0], str):
        ref_group_boundaries = []
        ori_cands, ori_refs = cands, refs
        cands, refs = [], []
        count = 0
        for cand, ref_group in zip(ori_cands, ori_refs):
            cands += [cand] * len(ref_group)
            refs += ref_group
            ref_group_boundaries.append((count, count + len(ref_group)))
            count += len(ref_group)

    if rescale_with_baseline:
        assert lang is not None, "Need to specify Language when rescaling with baseline"

    if model_type is None:
        lang = lang.lower()
        model_type = lang2model[lang]
    if num_layers is None:
        num_layers = model2layers[model_type]

    tokenizer = get_tokenizer(model_path, use_fast_tokenizer)
    model = get_model(model_path, num_layers, all_layers)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if not idf:
        idf_dict = defaultdict(lambda: 1.0)
        # set idf for [SEP] and [CLS] to 0
        idf_dict[tokenizer.sep_token_id] = 0
        idf_dict[tokenizer.cls_token_id] = 0
    elif isinstance(idf, dict):
        if verbose:
            print("using predefined IDF dict...")
        idf_dict = idf
    else:
        if verbose:
            print("preparing IDF dict...")
        start = time.perf_counter()
        idf_dict = get_idf_dict(refs, tokenizer, nthreads=nthreads)
        if verbose:
            print("done in {:.2f} seconds".format(time.perf_counter() - start))

    if verbose:
        print("calculating scores...")
    start = time.perf_counter()
    all_preds = bert_cos_score_idf(
        model,
        refs,
        cands,
        tokenizer,
        idf_dict,
        verbose=verbose,
        device=device,
        batch_size=batch_size,
        all_layers=all_layers,
    ).cpu()

    if ref_group_boundaries is not None:
        max_preds = []
        for beg, end in ref_group_boundaries:
            max_preds.append(all_preds[beg:end].max(dim=0)[0])
        all_preds = torch.stack(max_preds, dim=0)

    use_custom_baseline = baseline_path is not None
    if rescale_with_baseline:
        if baseline_path is None:
            baseline_path = os.path.join(
                os.path.dirname(__file__), f"rescale_baseline/{lang}/{model_type}.tsv"
            )
        if os.path.isfile(baseline_path):
            if not all_layers:
                baselines = torch.from_numpy(
                    pd.read_csv(baseline_path).iloc[num_layers].to_numpy()
                )[1:].float()
            else:
                baselines = (
                    torch.from_numpy(pd.read_csv(baseline_path).to_numpy())[:, 1:]
                    .unsqueeze(1)
                    .float()
                )

            all_preds = (all_preds - baselines) / (1 - baselines)
        else:
            print(
                f"Warning: Baseline not Found for {model_type} on {lang} at {baseline_path}",
                file=sys.stderr,
            )

    out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]  # P, R, F

    if verbose:
        time_diff = time.perf_counter() - start
        print(
            f"done in {time_diff:.2f} seconds, {len(refs) / time_diff:.2f} sentences/sec"
        )

    if return_hash:
        return tuple(
            [
                out,
                get_hash(
                    model_type,
                    num_layers,
                    idf,
                    rescale_with_baseline,
                    use_custom_baseline=use_custom_baseline,
                    use_fast_tokenizer=use_fast_tokenizer,
                ),
            ]
        )

    return out


def plot_example(
    candidate,
    reference,
    model_path=None,
    num_layers=None,
    lang=None,
    rescale_with_baseline=False,
    baseline_path=None,
    use_fast_tokenizer=False,
    fname="",
):
    """
    BERTScore metric.

    Args:
        - :param: `candidate` (str): a candidate sentence
        - :param: `reference` (str): a reference sentence
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `model_type` (str): bert specification, default using the suggested
                  model for the target langauge; has to specify at least one of
                  `model_type` or `lang`
        - :param: `num_layers` (int): the layer of representation to use
        - :param: `lang` (str): language of the sentences; has to specify
                  at least one of `model_type` or `lang`. `lang` needs to be
                  specified when `rescale_with_baseline` is True.
        - :param: `return_hash` (bool): return hash code of the setting
        - :param: `rescale_with_baseline` (bool): rescale bertscore with pre-computed baseline
        - :param: `use_fast_tokenizer` (bool): `use_fast` parameter passed to HF tokenizer
        - :param: `fname` (str): path to save the output plot
    """
    assert isinstance(candidate, str)
    assert isinstance(reference, str)
    model_type=os.path.basename(model_path)
    assert (
        lang is not None or model_path is not None
    ), "Either lang or model_type should be specified"

    if rescale_with_baseline:
        assert lang is not None, "Need to specify Language when rescaling with baseline"

    if model_type is None:
        lang = lang.lower()
        model_type = lang2model[lang]
    if num_layers is None:
        num_layers = model2layers[model_type]

    tokenizer = get_tokenizer(model_path, use_fast_tokenizer)
    model = get_model(model_path, num_layers)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    idf_dict = defaultdict(lambda: 1.0)
    # set idf for [SEP] and [CLS] to 0
    idf_dict[tokenizer.sep_token_id] = 0
    idf_dict[tokenizer.cls_token_id] = 0

    hyp_embedding, masks, padded_idf = get_bert_embedding(
        [candidate], model, tokenizer, idf_dict, device=device, all_layers=False
    )
    ref_embedding, masks, padded_idf = get_bert_embedding(
        [reference], model, tokenizer, idf_dict, device=device, all_layers=False
    )
    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))
    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    sim = sim.squeeze(0).cpu()

    # remove [CLS] and [SEP] tokens
    r_tokens = [tokenizer.decode([i]) for i in sent_encode(tokenizer, reference)][1:-1]
    h_tokens = [tokenizer.decode([i]) for i in sent_encode(tokenizer, candidate)][1:-1]
    sim = sim[1:-1, 1:-1]

    if rescale_with_baseline:
        if baseline_path is None:
            baseline_path = os.path.join(
                os.path.dirname(__file__), f"rescale_baseline/{lang}/{model_type}.tsv"
            )
        if os.path.isfile(baseline_path):
            baselines = torch.from_numpy(
                pd.read_csv(baseline_path).iloc[num_layers].to_numpy()
            )[1:].float()
            sim = (sim - baselines[2].item()) / (1 - baselines[2].item())
        else:
            print(
                f"Warning: Baseline not Found for {model_type} on {lang} at {baseline_path}",
                file=sys.stderr,
            )

    fig, ax = plt.subplots(figsize=(len(r_tokens), len(h_tokens)))
    im = ax.imshow(sim, cmap="Blues", vmin=0, vmax=1)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(r_tokens)))
    ax.set_yticks(np.arange(len(h_tokens)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(r_tokens, fontsize=10)
    ax.set_yticklabels(h_tokens, fontsize=10)
    ax.grid(False)
    plt.xlabel("Reference (tokenized)", fontsize=14)
    plt.ylabel("Candidate (tokenized)", fontsize=14)
    title = "Similarity Matrix"
    if rescale_with_baseline:
        title += " (after Rescaling)"
    plt.title(title, fontsize=14)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.2)
    fig.colorbar(im, cax=cax)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(h_tokens)):
        for j in range(len(r_tokens)):
            text = ax.text(
                j,
                i,
                "{:.3f}".format(sim[i, j].item()),
                ha="center",
                va="center",
                color="k" if sim[i, j].item() < 0.5 else "w",
            )

    fig.tight_layout()
    if fname != "":
        plt.savefig(fname, dpi=100)
        print("Saved figure to file: ", fname)
    plt.show()
