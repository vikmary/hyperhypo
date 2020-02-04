#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import ijson
import re
import argparse
from pathlib import Path
from random import sample
from collections import defaultdict
from datetime import datetime
from itertools import repeat
from typing import List, Dict, Union, Tuple, Optional

import hashedindex
import torch
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer

from hybert import HyBert
from dataset import HypoDataset
from train import device, to_device
from utils import get_test_senses, get_train_synsets, synsets2senses, get_wordnet_synsets
from prepare_corpora.utils import smart_open, Lemmatizer, Sanitizer
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
    parser.add_argument('--index-path', '-i', type=Path, required=True,
                        help='path to an index file,'
                        ' corpus with tokenized text should also be prebuilt')
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
                        k: int = 100,
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
                  score_hyperhypernym_synsets: bool = False,
                  wordnet_synsets: Optional[Dict] = None) -> List[Tuple[str, float]]:
    if pos and pos not in ('nouns', 'adjectives', 'verbs'):
        raise ValueError(f'Wrong value for pos \'{pos}\'.')
    synset_scores = defaultdict(list)
    for hyper, h_score in hypernym_preds:
        cand_synsets = hypernym2synsets[hyper]
        if score_hyperhypernym_synsets:
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
    def __init__(self,
                 index_path: Union[str, Path],
                 vocab: Optional[List[str]] = None) -> None:
        self.vocab = vocab
        self.index_path = Path(index_path)

        base_name = self.index_path.name.rsplit('.json', 1)[0]
        base_name = base_name.split('.', 2)[-1]
        if '-head-' in base_name:
            base_name, num_sents = base_name.rsplit('-head-', 1)
            num_sents = int(num_sents)
            print(f"Will load only {num_sents} sentences from corpus.")
        else:
            num_sents = sys.maxsize
        self.corpus_path = Path(self.index_path.with_name('corpus.' + base_name +
                                                          '.token.txt.gz'))
        if not self.corpus_path.exists():
            self.corpus_path = Path(self.index_path.with_name('corpus.' + base_name +
                                                              '.token.txt'))
        if not self.corpus_path.exists():
            raise RuntimeError(f"corpus {self.corpus_path} doesn't exists")

        self.idx = self.load_index(self.index_path, self.vocab)
        self.corpus = self.load_corpus(self.corpus_path, num_sents)

    @classmethod
    def load_corpus(cls,
                    path: Union[str, Path],
                    num_sents: int = sys.maxsize) -> List[str]:
        print(f"Loading corpus from {path}.", file=sys.stderr)
        with smart_open(path, 'rt') as fin:
            return [ln.strip() for _, ln in zip(range(num_sents), fin)]

    @classmethod
    def load_index(cls,
                   path: Union[str, Path],
                   vocab: Optional[List[str]] = None) -> hashedindex.HashedIndex:
        print(f"Loading index from {path}.", file=sys.stderr)
        index = hashedindex.HashedIndex()
        if vocab is not None:
            print(f"Will load only {len(vocab)} lemmas present in a vocab.")
        for lemma, idxs in ijson.kvitems(open(path, 'rt'), ''):
            if not vocab or (lemma in vocab):
                for idx in idxs:
                    index.add_term_occurrence(lemma, tuple(idx))
        return index

    def get_contexts(self,
                     lemma: str,
                     max_num_tokens: Optional[int] = None) -> List[Tuple[str, int]]:
        if lemma not in self.idx:
            print(f"Warning: lemma '{lemma}' not in index.", file=sys.stderr)
            return []
        sents = ((self.corpus[sent_idx].split(), pos)
                 for sent_idx, pos in self.idx.get_documents(lemma))
        if max_num_tokens is not None:
            sents = list(filter(lambda s_pos: len(s_pos[0]) < max_num_tokens, sents))
            if not sents:
                w_size = max_num_tokens // 2
                sents = ((self.corpus[s_id].split()[pos - w_size: pos + w_size],
                          w_size - max(w_size - pos, 0))
                         for s_id, pos in self.idx.get_documents(lemma))
        return list(sents)


if __name__ == "__main__":
    args = parse_args()

    sanitizer = Sanitizer(filter_stresses=True, filter_empty_brackets=True)
    tokenizer = re.compile(r"[\w']+|[^\w ]")
    lemmatizer = Lemmatizer()

    # load wordnet and word to get prediction for
    if args.is_train_format:
        test_synsets = get_train_synsets([args.data_path])
        test_senses = [s['content'].lower() for s in synsets2senses(test_synsets)]
    else:
        test_senses = [s['content'].lower() for s in get_test_senses([args.data_path])]
    test_lemmas = [' '.join(lemmatizer(t)
                            for t in tokenizer.findall(sanitizer(s).lower()))
                   for s in test_senses]

    # get corpus with contexts and it's index
    corpus = CorpusIndexed(args.index_path, vocab=test_lemmas)

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

    # generating output fila name
    now = datetime.now()
    out_pred_path = args.output_dir / ('bert.' + args.data_path.name.rstrip('.tsv') +
                                       f'_pred_d{now.strftime("%Y%m%d_%H:%M")}.tsv')
    print(f"Writing predictions to {out_pred_path}.")

    n_skipped = 0
    with open(out_pred_path, 'wt') as f_pred:
        for word, lemma in tqdm(zip(test_senses, test_lemmas)):
            contexts = corpus.get_contexts(lemma, max_num_tokens=250)
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
                                             score_hyperhypernym_synsets=False)
            for s_id, score in pred_synsets:
                h_senses_str = ','.join(sense['content']
                                        for sense in synsets[s_id]['senses'])
                f_pred.write(f'{word.upper()}\t{s_id}\t{score}\t{h_senses_str}\n')
    print(f"Wrote predictions to {out_pred_path}.")
    print(f"Skipped {n_skipped}/{len(test_senses)}"
          f" ({int(n_skipped/len(test_senses) * 100)} %) test words.")
