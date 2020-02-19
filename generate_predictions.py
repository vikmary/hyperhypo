#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import json
import argparse
import itertools
import collections
import datetime
from pathlib import Path
from random import sample
from typing import List, Dict, Union, Tuple, Optional

import hashedindex
import torch
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer

from hybert import HyBert
from dataset import HypoDataset, get_indices_and_masks
from embedder import device, to_device
from postprocess_prediction import get_prediction
from corpus_indexed import CorpusIndexed
from utils import get_test_senses, get_train_synsets, synsets2senses, get_wordnet_synsets
from utils import enrich_with_wordnet_relations
from prepare_corpora.utils import TextPreprocessor


def parse_args():
    parser = argparse.ArgumentParser()
    # Test words parameters
    parser.add_argument('--data-path', type=Path, required=True,
                        help='dataset path to get predictions for')
    parser.add_argument('--synset-info-paths', type=Path, nargs='+',
                        help='paths to synset info for data if is-train-format')
    parser.add_argument('--pos', type=str, default=None,
                        help='filter hypernyms of only this type of pos')
    parser.add_argument('--is-train-format', '-T', action='store_true',
                        help='whether input data is in train format or in test format')
    parser.add_argument('--synset-level', '-S', action='store_true',
                        help='predict from model on the level of synsets,'
                        ' not on the level of words')
    # Data required for making predictions
    parser.add_argument('--wordnet-dir', type=Path, required=True,
                        help='path to a wordnet directory')
    parser.add_argument('--corpus-path', type=Path,
                        help='path to a tokenized corpus, lemmatized corpus'
                        ' should be also prebuilt')
    parser.add_argument('--candidates', type=Path, required=True,
                        help='path to a list of candidates')
    parser.add_argument('--fallback-prediction-path', '-f', type=Path,
                        help='file with predictions to fallback to')
    # Model parameters
    parser.add_argument('--bert-model-dir', type=Path, required=True,
                        help='path to a trained bert directory')
    parser.add_argument('--load-checkpoint', type=Path,
                        help='path to a pytorch checkpoint')
    parser.add_argument('--num-contexts', default=1, type=int,
                        help='number of averaged contexts for each hyponym')
    parser.add_argument('--batch-size', default=2, type=int,
                        help='batch size for inferrinf predictions')
    parser.add_argument('--max-context-length', default=512, type=int,
                        help='maximum length of context in subtokens')
    parser.add_argument('--metric', default='product', choices=('product', 'cosine'),
                        help='metric to use for choosing the best predictions')
    parser.add_argument('--do-not-mask-hypernym', action='store_true',
                        help='represent hypernyms with whole contexts, do not mask'
                        ' other words')
    # Ouput path
    parser.add_argument('--output-prefix', '-o', type=str, required=True,
                        help='path to a file with it\'s prefix')
    return parser.parse_args()


def predict_with_hybert(model: HyBert,
                        contexts: List[Tuple[List[str], int, int]],
                        k: int = 30,
                        metric: str = 'product',
                        batch_size: int = 4,
                        mask_hypernym: bool = True,
                        max_length: int = 512) -> List[Tuple[List[str], float]]:
    if metric not in ('product', 'cosine'):
        raise ValueError(f'metric parameter has invalid value {metric}')

    hypernym_repr_t = []
    for b_start_id in range(0, len(contexts), batch_size):
        b_subtok_idxs, b_hypo_masks = [], []
        for context, l_start, l_end in contexts[b_start_id: b_start_id + batch_size]:
            if not mask_hypernym:
                l_start, l_end = 0, len(context)
            subtok_idxs, hypo_mask, subtok_start, subtok_end = \
                get_indices_and_masks(context, l_start, l_end, model.tokenizer)
            subtok_idxs, hypo_mask, subtok_start, subtok_end = \
                HypoDataset._cut_to_maximum_length(subtok_idxs,
                                                   hypo_mask,
                                                   subtok_start,
                                                   subtok_end,
                                                   length=max_length - 2)
            cls_idx, sep_idx = model.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
            b_subtok_idxs.append([cls_idx] + subtok_idxs + [sep_idx])
            b_hypo_masks.append([0.0] + hypo_mask + [0.0])
        batch = HypoDataset.torchify_and_pad(b_subtok_idxs, b_hypo_masks)

        print(batch[0].shape)
        # print('indices', batch[0])
        # print('hypo_mask', batch[1])
        # print('attn_mask', batch[2])
        # hypernym_repr_t[-1]: [batch_size, hidden_size]
        with torch.no_grad():
            hypernym_repr_t.append(model(*to_device(*batch))[0])
    # hypernym_repr_t: [num_contexts, hidden_size]
    hypernym_repr_t = torch.cat(hypernym_repr_t, dim=0)
    # hypernym_logits_t: [num_contexts, vocab_size]
    if metric == 'product':
        hypernym_logits_t = hypernym_repr_t @ model.hypernym_embeddings.T
    if metric == 'cosine':
        # dot product of normalized vectors is equivalent to cosine
        hypernym_repr_t /= hypernym_repr_t.norm(dim=0, keepdim=True)
        hypernym_embeddings_norm_t = model.hypernym_embeddings /\
            model.hypernym_embeddings.norm(dim=1, keepdim=True)
        hypernym_logits_t = hypernym_repr_t @ hypernym_embeddings_norm_t.T
    hypernym_logits_t = torch.log_softmax(hypernym_logits_t, dim=1)
    # hypernym_logits_avg_t: [hidden_size]
    hypernym_logits_avg_t = hypernym_logits_t.mean(dim=0)
    hypernym_logits = hypernym_logits_avg_t.cpu().detach().numpy()
    return sorted(zip(model.hypernym_list, hypernym_logits),
                  key=lambda h_sc: h_sc[1], reverse=True)[:k]


def rescore_synsets(hypernym_preds: List[Tuple[Union[List[str], str], float]],
                    pos: Optional[str] = None,
                    by: str = 'max',
                    k: int = 10,
                    score_hyperhypernym_synsets: bool = False,
                    wordnet_synsets: Optional[Dict] = None) -> List[Tuple[str, float]]:
    if pos and pos not in ('nouns', 'adjectives', 'verbs'):
        raise ValueError(f'Wrong value for pos \'{pos}\'.')
    if by == 'mean':
        aggr_fn = lambda scores: sum(scores) / len(scores)
    elif by == 'max':
        aggr_fn = lambda scores: max(scores)
    elif by == 'sum':
        aggr_fn = lambda scores: sum(math.exp(s) for s in scores)
    else:
        raise ValueError(f'Wrong value for by \'{by}\'')

    synset_scores = collections.defaultdict(list)
    for cand_synsets, h_score in hypernym_preds:
        if isinstance(cand_synsets, str):
            cand_synsets = [cand_synsets]
        if score_hyperhypernym_synsets:
            cand_synsets = [h_s['id']
                            for s in cand_synsets
                            for h_s in (wordnet_synsets[s].get('hypernyms') or [{'id': s}])]
        for h_synset in cand_synsets:
            if pos and (h_synset[-1].lower() != pos[0]):
                continue
            synset_scores[h_synset].append(h_score)
    synset_aggr_scores = {synset: aggr_fn(scores)
                          for synset, scores in synset_scores.items()}
    return sorted(synset_aggr_scores.items(), key=lambda x: x[1], reverse=True)[:k]


def load_candidates(fname: Union[str, Path],
                    senses2synset: bool = False) -> Dict[str, List[str]]:
    cands = collections.defaultdict(list)
    with open(fname, 'rt') as fin:
        for row in fin:
            cand_word, cand_synset_id = row.split('\t', 2)[:2]
            if senses2synset:
                cands[cand_synset_id].append(cand_word)
            else:
                cands[cand_word].append(cand_synset_id)
    if senses2synset:
        return {tuple(senses): synset_id for synset_id, senses in cands.items()}
    return cands


if __name__ == "__main__":
    args = parse_args()

    preprocessor = TextPreprocessor('regexp+pymorphy',
                                    filter_stresses=True,
                                    filter_empty_brackets=True,
                                    lemmatize=True,
                                    lowercase=True)

    print(f"Loading BertModel from {args.bert_model_dir}.")
    config = BertConfig.from_pretrained(args.bert_model_dir / 'bert_config.json')
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir, do_lower_case=False)
    bert = BertModel.from_pretrained(str(args.bert_model_dir / 'ptrubert.pt'),
                                     config=config)
    bert.to(device)

    print(f"Initializing HyBert.")
    if args.synset_level:
        candidates = load_candidates(args.candidates, senses2synset=True)
        model = HyBert(bert, tokenizer, list(candidates.keys()), True)
    else:
        candidates = load_candidates(args.candidates)
        model = HyBert(bert, tokenizer, [[k] for k in candidates], True)
    model.to(device)
    if args.load_checkpoint:
        print(f"Loading HyBert from {args.load_checkpoint}.")
        model_state = model.state_dict()
        model_state.update({k: v for k, v in torch.load(args.load_checkpoint,
                                                        map_location=device).items()
                            if k != 'hypernym_embeddings'})
    else:
        print(f"Initializing Hybert from ruBert.")
    model.eval()
    print(f"Batch size equals {args.num_contexts}.")
    print(f"Scoring candidates with '{args.metric}' metric.")

    # load wordnet and word to get prediction for
    if args.is_train_format:
        test_synsets = get_train_synsets([args.data_path], args.synset_info_fpaths)
        test_senses = [s['content'].lower() for s in synsets2senses(test_synsets)]
    else:
        test_senses = [s['content'].lower() for s in get_test_senses([args.data_path])]
    # test_senses = ['ЭПИЛЕПСИЯ', 'ЭЯКУЛЯЦИЯ', 'ЭПОЛЕТ']
    test_lemmas = [preprocessor(s)[1] for s in test_senses]

    # get corpus with contexts and it's index
    if args.corpus_path is not None:
        print("Embedding using corpora.")
        index_path = CorpusIndexed.get_index_path(args.corpus_path,
                                                  suffix=args.data_path.stem)
        print(f'Index path : {index_path}.')
        if not index_path.exists():
            lemma_corpus_path = CorpusIndexed.get_corpus_path(index_path, level='lemma')
            print(f'Lemmatized corpus path : {lemma_corpus_path}.')
            index = CorpusIndexed.build_index(lemma_corpus_path,
                                              vocab=test_lemmas,
                                              max_utterances_per_item=args.num_contexts)
            json.dump(index, open(index_path, 'wt'), indent=2, ensure_ascii=False)
        corpus = CorpusIndexed.from_index(index_path, vocab=test_lemmas)
    else:
        print("Embedding words without context.")
        corpus = None

    synsets = get_wordnet_synsets(args.wordnet_dir.glob('synsets.*'))
    enrich_with_wordnet_relations(synsets, args.wordnet_dir.glob('synset_relations.*'))

    # load fallback predictions if needed
    fallback_preds = []
    if args.fallback_prediction_path:
        fallback_preds = {w: preds
                          for w, preds in get_prediction(args.fallback_prediction_path)}

    # generating output file name
    now = datetime.datetime.now()
    out_pred_path = args.output_prefix + '.' + args.data_path.name.rstrip('.tsv') +\
        f'_pred_d{now.strftime("%Y%m%d_%H:%M")}.tsv'
    print(f"Writing predictions to {out_pred_path}.")

    n_skipped = 0
    with open(out_pred_path, 'wt') as f_pred:
        for word, lemma in tqdm(zip(test_senses, test_lemmas), total=len(test_senses)):
            # import ipdb; ipdb.set_trace()
            contexts = []
            if corpus:
                contexts = corpus.get_contexts(lemma) or contexts
            if not contexts:
                n_skipped += 1
                if fallback_preds:
                    if word in fallback_preds:
                        pred_synsets = zip(fallback_preds[word], itertools.repeat('nan'))
                    else:
                        contexts = contexts or [(word.lower().split(), 0, len(word.split()))]
                        print(f"Warning: {word} not in fallback_predictions")
                else:
                    contexts = contexts or [(word.lower().split(), 0, len(word.split()))]
            if contexts:
                random_contexts = sample(contexts, min(args.num_contexts, len(contexts)))
                print(f"Random context ({word}) = {random_contexts[0]}")
                try:
                    preds = predict_with_hybert(model,
                                                random_contexts,
                                                metric=args.metric,
                                                k=30,
                                                mask_hypernym=not args.do_not_mask_hypernym,
                                                batch_size=args.batch_size,
                                                max_length=args.max_context_length)
                except Exception as msg:
                    print(f"captured exception with msg = '{msg}'")
                    import ipdb; ipdb.set_trace()
                if args.synset_level:
                    pred_synsets = [([candidates[h]], sc) for h, sc in preds]
                    pred_senses = [([s['content'] for s in synsets[p[0]]['senses']], sc)
                                   for p, sc in pred_synsets]
                    print(f"Pred synsets ({word}): {pred_senses[:4]}")
                else:
                    pred_synsets = [(candidates[h[0]], sc) for h, sc in preds]
                    print(f"Pred hyponyms ({word}): {preds[:4]}")
                pred_synsets = rescore_synsets(pred_synsets,
                                               by='sum',
                                               pos=args.pos,
                                               wordnet_synsets=synsets,
                                               score_hyperhypernym_synsets=True)
                pred_senses = [([s['content'] for s in synsets[p]['senses']], sc)
                               for p, sc in pred_synsets]
                print(f"Rescored pred synsets({word}): {pred_senses[:4]}")
            for s_id, score in pred_synsets:
                h_senses_str = ','.join(sense['content']
                                        for sense in synsets[s_id]['senses'])
                f_pred.write(f'{word.upper()}\t{s_id}\t{score}\t{h_senses_str}\n')
    print(f"Wrote predictions to {out_pred_path}.")
    print(f"{n_skipped}/{len(test_senses)} ({int(n_skipped/len(test_senses)*100)}%)"
          f" test words are without context.")
