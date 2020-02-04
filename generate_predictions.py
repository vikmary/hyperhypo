#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import argparse
from pathlib import Path
from random import sample
from collections import defaultdict
from datetime import datetime
from itertools import repeat
from typing import List, Dict, Union, Tuple, Optional

import torch
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer

from hybert import HyBert
from dataset import HypoDataset
from train import device, to_device
from utils import get_test_senses, get_train_synsets, synsets2senses, get_wordnet_synsets
from postprocess_prediction import get_prediction


def parse_args():
    parser = argparse.ArgumentParser()
    # Test words parameters
    parser.add_argument('--data-path', '-d', type=Path, required=True,
                        help='dataset path to get predictions for')
    parser.add_argument('--pos', type=str, default=None,
                        help='filter hypernyms of only this type of pos')
    parser.add_argument('--is-train-format', '-T', action='store_true',
                        help='whether input data is in train format or in test format')
    # Data required for making predictions
    parser.add_argument('--wordnet-dir', '-w', type=Path, required=True,
                        help='path to a wordnet directory')
    parser.add_argument('--corpus-path', '-t', type=Path, required=True,
                        help='path to a corpus file, index file should also be prebuilt')
    parser.add_argument('--candidates', '-c', type=Path, required=True,
                        help='path to a list of candidates')
    parser.add_argument('--fallback-prediction-path', '-f', type=Path,
                        help='file with predictions to fallback to')
    # Model path
    parser.add_argument('--bert-model-dir', '-b', type=Path, required=True,
                        help='path to a trained bert directory')
    parser.add_argument('--load-checkpoint', '-l', type=Path,
                        help='path to a pytorch checkpoint')
    # Ouput path
    parser.add_argument('--output_dir', '-o', type=Path, required=True,
                        help='output directory for labels prepared for scoring')
    return parser.parse_args()


def predict_with_hybert(model: HyBert,
                        context: List[str],
                        hyponym_pos: int,
                        k: int = 1000,
                        metric: str = 'product') -> List[str]:
    if metric not in ('product', 'cosine'):
        raise ValueError(f'metric parameter has invalid value {metric}')
    subtoken_idxs, hyponym_mask = [], []
    for i, token in enumerate(['[CLS]'] + context + ['[SEP]']):
        subtokens = model.tokenizer.tokenize(token)
        subtoken_idxs.extend(model.tokenizer.convert_tokens_to_ids(subtokens))
        hyponym_mask.extend([float(i == pos + 1)] * len(subtokens))
    batch = HypoDataset.torchify_and_pad([subtoken_idxs], [hyponym_mask])

    hypernym_logits = model(*to_device(*batch)).cpu().detach().numpy()[0]
    if metric == 'cosine':
        # TODO: try cosine here
        raise NotImplementedError()
    return sorted(zip(model.hypernym_list, hypernym_logits),
                  key=lambda h_sc: h_sc[1], reverse=True)[:k]


def score_synsets(hypernym_preds: List[Tuple[str, float]],
                  hypernym2synsets: Dict[str, List[str]],
                  pos: Optional[str] = None,
                  k: int = 20,
                  score_hypernym_synsets: bool = False,
                  wordnet_synsets: Optional[Dict] = None) -> List[Tuple[str, float]]:
    if pos and pos not in ('nouns', 'adjectives', 'verbs'):
        raise ValueError(f'Wrong value for pos \'{pos}\'.')
    synset_scores = defaultdict(list)
    for hyper, h_score in hypernym_preds:
        cand_synsets = hypernym2synsets[hyper]
        if score_hypernym_synsets:
            cand_synsets = [h_s['id']
                            for s in cand_synsets
                            for h_s in wordnet_synsets[s].get('hypernyms', [{'id': s}])]
        for h_synset in cand_synsets:
            if pos and (h_synset[-1].lower() != pos[0]):
                continue
            synset_scores[h_synset].append(h_score)
    synset_mean_scores = {synset: sum(scores)/len(scores)
                          for synset, scores in synset_scores.items()}
    return sorted(synset_mean_scores.items(), key=lambda x: x[1], reverse=True)[:k]


def load_candidates(fname: Union[str, Path]) -> Dict[str, List[str]]:
    cands = defaultdict(list)
    with open(fname, 'rt') as fin:
        for row in fin:
            cand_word, cand_synset_id = row.split('\t', 2)[:2]
            cands[cand_word].append(cand_synset_id)
    return cands


class CorpusIndexed:
    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        base_name = self.path.name.rsplit('.token')[0][7:]
        self.index_path = self.path.with_name('index.full.' + base_name + '.json')
        if not self.index_path.exists():
            raise RuntimeError(f"index {self.index_path} doesn't exists")
        if not self.path.exists():
            raise RuntimeError(f"corpus {self.path} doesn't exists")

        print(f"Loading corpus from {self.path}.", file=sys.stderr)
        self.corpus = [ln.strip() for ln in open(self.path, 'rt')]
        print(f"Loading index from {self.index_path}.", file=sys.stderr)
        self.idx = {w: idxs for w, idxs in json.load(open(self.index_path, 'rt')).items()}

    def get_contexts(self,
                     word: str,
                     max_num_tokens: Optional[int] = None) -> List[Tuple[str, int]]:
        if word not in self.idx:
            print(f"Warning: word '{word}' not in index.", file=sys.stderr)
            return []
        sents = ((self.corpus[sent_idx].split(), pos) for sent_idx, pos in self.idx[word])
        if max_num_tokens is not None:
            sents = list(filter(lambda s_pos: len(s_pos[0]) < max_num_tokens, sents))
            if not sents:
                w_size = max_num_tokens // 2
                sents = ((self.corpus[s_id].split()[pos - w_size: pos + w_size],
                          w_size - max(w_size - pos, 0))
                         for s_id, pos in self.idx[word])
        return list(sents)


if __name__ == "__main__":
    args = parse_args()

    # load wordnet and word to get prediction for
    if args.is_train_format:
        test_synsets = get_train_synsets([args.data_path])
        test_senses = [s['content'].lower() for s in synsets2senses(test_synsets)]
    else:
        test_senses = [s['content'].lower() for s in get_test_senses([args.data_path])]
    synsets = get_wordnet_synsets(args.wordnet_dir.glob('synsets.*'))

    # load fallback predictions if needed
    fallback_preds = []
    if args.fallback_prediction_path:
        fallback_preds = {w: preds
                          for w, preds in get_prediction(args.fallback_prediction_path)}

    # load Bert model
    config = BertConfig.from_pretrained(args.bert_model_dir / 'bert_config.json')
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir, do_lower_case=False)
    bert = BertModel.from_pretrained(str(args.bert_model_dir / 'ptrubert.pt'),
                                     config=config)

    candidates = load_candidates(args.candidates)
    model = HyBert(bert, tokenizer, list(candidates.keys()))
    model.to(device)
    if args.load_checkpoint:
        print(f"Loading HyBert from {args.load_checkpoint}.")
        model_state = model.state_dict()
        model_state.update({k: v for k, v in torch.load(args.load_checkpoint).items()
                            if k != 'hypernym_embeddings'})
    else:
        print(f"Initializing Hybert from ruBert.")
    model.eval()

    # get corpus with contexts and it's index
    corpus = CorpusIndexed(args.corpus_path)

    # generating output fila name
    now = datetime.now()
    out_pred_path = args.output_dir / ('bert.' + args.data_path.name.rstrip('.tsv') +
                                       f'_pred_d{now.strftime("%Y%m%d_%H:%M")}.tsv')
    print(f"Writing predictions to {out_pred_path}.")

    n_skipped = 0
    with open(out_pred_path, 'wt') as f_pred:
        for word in tqdm(test_senses):
            contexts = corpus.get_contexts(word, max_num_tokens=250)
            if not contexts:
                n_skipped += 1
                if word not in fallback_preds:
                    pred_synsets = [(sample(synsets.keys(), 1)[0], 'nan')]
                    if not fallback_preds:
                        print(f"Warning: {word} not in fallback_predictions")
                else:
                    pred_synsets = zip(fallback_preds[word], repeat('nan'))
            else:
                random_context, pos = sample(contexts, 1)[0]
                print(f"Random context ({word}) = {random_context}, pos = {pos}")
                pred_hypernyms = predict_with_hybert(model, random_context, pos)
                print(f"Pred hypernyms ({word}): {pred_hypernyms[:2]}")
                pred_synsets = score_synsets(pred_hypernyms,
                                             candidates,
                                             pos=args.pos,
                                             wordnet_synsets=synsets,
                                             score_hypernym_synsets=True)
            for s_id, score in pred_synsets:
                h_senses_str = ','.join(sense['content']
                                        for sense in synsets[s_id]['senses'])
                f_pred.write(f'{word.upper()}\t{s_id}\t{score}\t{h_senses_str}\n')
    print(f"Wrote predictions to {out_pred_path}.")
    print(f"Skipped {n_skipped}/{len(test_senses)}"
          f" ({int(n_skipped/len(test_senses) * 100)} %) test words.")
