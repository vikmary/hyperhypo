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
from wiktionary import DefinitionDB
from utils import get_test_senses, get_train_synsets, synsets2senses, get_wordnet_synsets
from utils import enrich_with_wordnet_relations, get_all_related
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
    parser.add_argument('--embed-with-ruthes-name', action='store_true',
                        help='embed hypernym synsets with ruthes names')
    # Data required for making predictions
    parser.add_argument('--wordnet-dir', type=Path, required=True,
                        help='path to a wordnet directory')
    parser.add_argument('--corpus-path', type=Path,
                        help='path to a tokenized corpus, lemmatized corpus'
                        ' should be also prebuilt')
    parser.add_argument('--use-definitions', action='store_true',
                        help='whether to embed hyponyms with definitions')
    parser.add_argument('--candidates', type=Path, required=True,
                        help='path to a list of candidates')
    parser.add_argument('--fallback-prediction-path', '-f', type=Path,
                        help='file with predictions to fallback to')
    # Model parameters
    parser.add_argument('--bert-model-dir', type=Path, required=True,
                        help='path to a trained bert directory')
    parser.add_argument('--load-checkpoint', type=Path,
                        help='path to a pytorch checkpoint')
    parser.add_argument('--use-projection', action='store_true',
                        help='construct projection output layer')
    parser.add_argument('--num-contexts', default=1, type=int,
                        help='number of averaged contexts for each hyponym')
    parser.add_argument('--batch-size', default=2, type=int,
                        help='batch size for inferrinf predictions')
    parser.add_argument('--max-context-length', default=512, type=int,
                        help='maximum length of context in subtokens')
    parser.add_argument('--metric', default='product', choices=('product', 'cosine'),
                        help='metric to use for choosing the best predictions')
    parser.add_argument('--embed-with-context', action='store_true',
                        help='represent hypernyms with whole contexts, do not mask out'
                        ' other words')
    parser.add_argument('--embed-with-special-tokens', action='store_true',
                        help='use [CLS] and [SEP] when embedding phrases')
    parser.add_argument('--score-hyperhyper', action='store_true',
                        help='get hypernyms of predictions and score them')
    parser.add_argument('--one-per-group', action='store_true',
                        help='group predicted synsets and output one prediction'
                        'for each group')
    # Ouput path
    parser.add_argument('--output-prefix', '-o', type=str, required=True,
                        help='path to a file with it\'s prefix')
    return parser.parse_args()


def predict_with_hybert(model: HyBert,
                        contexts: List[Tuple[List[str], int, int]],
                        k: int,
                        metric: str,
                        batch_size: int,
                        embed_with_context: bool = False,
                        embed_with_special_tokens: bool = False,
                        max_length: int = 512) -> List[Tuple[List[str], float]]:
    if metric not in ('product', 'cosine'):
        raise ValueError(f'metric parameter has invalid value {metric}')

    hypernym_repr_t = []
    for b_start_id in range(0, len(contexts), batch_size):
        b_subtok_idxs, b_hypo_masks = [], []
        for context, l_start, l_end in contexts[b_start_id: b_start_id + batch_size]:
            subtok_idxs, hypo_mask, subtok_start, subtok_end = \
                get_indices_and_masks(context, l_start, l_end, model.tokenizer)
            subtok_idxs, hypo_mask, subtok_start, subtok_end = \
                HypoDataset._cut_to_maximum_length(subtok_idxs,
                                                   hypo_mask,
                                                   subtok_start,
                                                   subtok_end,
                                                   length=max_length - 2)
            if embed_with_context:
                hypo_mask = [1.0] * len(subtok_idxs)
            cls_idx, sep_idx = model.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
            b_subtok_idxs.append([cls_idx] + subtok_idxs + [sep_idx])
            if embed_with_special_tokens:
                b_hypo_masks.append([1.0] + hypo_mask + [1.0])
            else:
                b_hypo_masks.append([0.0] + hypo_mask + [0.0])
        batch = HypoDataset.torchify_and_pad(b_subtok_idxs, b_hypo_masks)

        print(batch[0].shape)
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
    hypernym_probs_t = torch.softmax(hypernym_logits_t, dim=1)
    # hypernym_probs_avg_t: [hidden_size]
    hypernym_probs_avg_t = hypernym_probs_t.mean(dim=0)
    hypernym_probs = hypernym_probs_avg_t.cpu().detach().numpy()
    return sorted(zip(model.hypernym_list, hypernym_probs),
                  key=lambda h_sc: h_sc[1], reverse=True)[:k]


def group_synsets(synsets: List[str], wordnet_synsets: Dict) -> Dict[str, str]:
    syn2group = {}
    for s_id in synsets:
        if s_id not in syn2group:
            syn2group[s_id] = get_all_related(s_id, wordnet_synsets, ('hypernyms',))
            syn2group[s_id] = {k: level
                               for k, level in syn2group[s_id].items()
                               if k in synsets}
    # print('syn2group', syn2group)
    not_covered = set(syn2group.keys())
    syn2root_group = {}
    for s_i_id in syn2group:
        if s_i_id not in not_covered:
            continue
        not_covered.remove(s_i_id)
        group_ids, root_group = [s_i_id], syn2group[s_i_id]
        for s_j_id in syn2root_group:
            if syn2root_group[s_j_id].keys() & root_group.keys():
                group_ids.append(s_j_id)
                if len(syn2root_group[s_j_id]) > len(root_group):
                    root_group = syn2group[s_j_id]
        for s_j_id in not_covered:
            if syn2group[s_j_id].keys() & root_group.keys():
                group_ids.append(s_j_id)
                if len(syn2group[s_j_id]) > len(root_group):
                    root_group = syn2group[s_j_id]
        for s_id in group_ids:
            if s_id in not_covered:
                not_covered.remove(s_id)
            syn2root_group[s_id] = root_group
        # print('syn2root_group', syn2root_group)
    syn2root_syn = {s_id: max(root_group, key=lambda s: root_group[s])
                    for s_id, root_group in syn2root_group.items()}

    # print('syn2root_syn', syn2root_syn)
    return syn2root_syn


def rescore_synsets(hypernym_preds: List[Tuple[Union[List[str], str], float]],
                    by: str,
                    k: int,
                    score_hyperhypernym_synsets: bool,
                    one_per_group: bool,
                    pos: Optional[str] = None,
                    wordnet_synsets: Optional[Dict] = None) -> List[Tuple[str, float]]:
    if pos and pos not in ('nouns', 'adjectives', 'verbs'):
        raise ValueError(f'Wrong value for pos \'{pos}\'.')
    if by == 'mean':
        aggr_fn = lambda scores: sum(scores) / len(scores)
    elif by == 'max':
        aggr_fn = lambda scores: max(scores)
    elif by == 'sum':
        aggr_fn = lambda scores: sum(scores)
    else:
        raise ValueError(f'Wrong value for by \'{by}\'')

    hypernym_preds_new = []
    for cand_synsets, h_score in hypernym_preds:
        if isinstance(cand_synsets, str):
            cand_synsets = [cand_synsets]
        if score_hyperhypernym_synsets:
            cand_synsets = [h_s['id']
                            for s in cand_synsets
                            for h_s in (wordnet_synsets[s].get('hypernyms') or
                                        [{'id': s}])]
        if pos:
            cand_synsets = [s_id for s_id in cand_synsets if (s_id[-1].lower() == pos[0])]
        hypernym_preds_new.append((cand_synsets, h_score))

    synset_scores = collections.defaultdict(list)
    for cand_synsets, h_score in hypernym_preds_new:
        for h_synset in cand_synsets:
            synset_scores[h_synset].append(h_score)
    synset_aggr_scores = {synset: aggr_fn(scores)
                          for synset, scores in synset_scores.items()}

    if one_per_group:
        uniq_synsets = set(s_id for s_ids, _ in hypernym_preds_new for s_id in s_ids)
        syn2group_syn = group_synsets(list(uniq_synsets), wordnet_synsets)
        syn_str = [(wordnet_synsets[k]['ruthes_name'], wordnet_synsets[v]['ruthes_name'])
                   for k, v in syn2group_syn.items()]
        print(f"Found {len(set(syn2group_syn.values()))} unique groups of {len(uniq_synsets)} predicted synsets.")
        print(f'syn2group_syn map = {syn_str}.')

        group_scores = collections.defaultdict(list)
        for s_id, score in synset_aggr_scores.items():
            group_scores[syn2group_syn[s_id]].append(score)
        group_aggr_scores = {gr: max(scores) for gr, scores in group_scores.items()}
        return sorted(group_aggr_scores.items(), key=lambda x: x[1], reverse=True)[:k]
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

    synsets = get_wordnet_synsets(args.wordnet_dir.glob('synsets.*'))
    enrich_with_wordnet_relations(synsets, args.wordnet_dir.glob('synset_relations.*'))

    print(f"Loading BertModel from {args.bert_model_dir}.")
    config = BertConfig.from_pretrained(args.bert_model_dir / 'bert_config.json')
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir, do_lower_case=False)
    bert = BertModel.from_pretrained(str(args.bert_model_dir / 'ptrubert.pt'),
                                     config=config)
    bert.to(device)

    print(f"Initializing HyBert.")
    if args.synset_level:
        candidates = load_candidates(args.candidates, senses2synset=True)
        if args.embed_with_ruthes_name:
            candidates = {(synsets[s_id]['ruthes_name'],): s_id
                          for s_id in candidates.values()}
        hypernym_list = list(candidates.keys())
    else:
        if args.embed_with_ruthes_name:
            raise ValueError(f'embedding with ruthes name is not supported for phrase '
                             f'level')
        candidates = load_candidates(args.candidates)
        hypernym_list = [[k] for k in candidates]
    model = HyBert(bert, tokenizer, hypernym_list,
                   use_projection=args.use_projection,
                   embed_wo_special_tokens=not args.embed_with_special_tokens,
                   embed_with_encoder_output=True)
    model.to(device)
    if args.load_checkpoint:
        print(f"Loading HyBert from {args.load_checkpoint}.")
        model_state = model.state_dict()
        model_state.update({k: v for k, v in torch.load(args.load_checkpoint,
                                                        map_location=device).items()
                            if k != 'hypernym_embeddings'})
        model.load_state_dict(model_state)
    else:
        print(f"Initializing Hybert from ruBert.")
    model.eval()
    print(f"Batch size equals {args.num_contexts}.")
    print(f"Scoring candidates with '{args.metric}' metric.")

    # load wordnet and word to get prediction for
    test_senses = [s['content'].lower() for s in get_test_senses([args.data_path])]
    # test_senses = ['ЭПИЛЕПСИЯ', 'ЭЯКУЛЯЦИЯ', 'ЭПОЛЕТ']
    test_lemmas = [preprocessor(s)[1] for s in test_senses]

    # get corpus with contexts and it's index
    corpus = None
    if args.use_definitions:
        print("Embeding using definitions.")
        ddb = DefinitionDB()
        # if not args.embed_with_context:
        #     raise NotImplementedError('embedding wo countext is not availabled'
        #                               ' for embedding with definitions right now.')
        if args.corpus_path is not None:
            raise ValueError('corpus can\'t be used if definitions are used')
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
            embed_with_context = args.embed_with_context
            embed_with_special_tokens = args.embed_with_special_tokens
            contexts = []
            if args.use_definitions:
                definitions = ddb(word)
                if definitions != [word]:
                    # def_tokens = word.split() + ['—'] + definition.split()
                    contexts = []
                    for d in definitions:
                        pos_start, pos_end = None, None
                        d_tokens = d.split()
                        word_lower_tokens = word.lower().split()
                        d_lower_tokens = [tok.lower() for tok in d_tokens]
                        if word_lower_tokens[0] in d_lower_tokens:
                            pos_start = d_lower_tokens.index(word_lower_tokens[0])
                            if d_lower_tokens[pos_start:pos_start +
                                              len(word_lower_tokens)] == \
                                    word_lower_tokens:
                                pos_end = pos_start + len(word_lower_tokens)
                        if pos_end is None:
                            if '-' in d_tokens:
                                pos_start, pos_end = 0, d_tokens.index('-')
                            elif '—' in d_tokens:
                                pos_start, pos_end = 0, d_tokens.index('—')
                            else:
                                word_tokens = (word[0].upper() + word[1:]).split()
                                d_tokens = word_tokens + ['—'] + d_tokens
                                pos_start, pos_end = 0, len(word_tokens)

                        contexts.append((d_tokens, pos_start, pos_end))
                    # contexts = [(d.split(), 0, 1) for d in definitions if d.strip()]
                    # contexts = [(def_tokens, 0, len(word.split()))]
                    # embed_with_context = True
                    # embed_with_special_tokens = True
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
                if args.use_definitions:
                    print(f"Contexts ({word}) = {random_contexts}")
                else:
                    print(f"Random context ({word}) = {random_contexts[0]}")
                try:
                    preds = predict_with_hybert(model,
                                                random_contexts,
                                                metric=args.metric,
                                                k=100,
                                                embed_with_context=embed_with_context,
                                                embed_with_special_tokens=embed_with_special_tokens,
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
                                               k=15,
                                               pos=args.pos,
                                               wordnet_synsets=synsets,
                                               one_per_group=args.one_per_group,
                                               score_hyperhypernym_synsets=args.score_hyperhyper)
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
