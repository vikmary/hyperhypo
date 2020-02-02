#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from itertools import chain
from pathlib import Path
from typing import Union, List, Tuple, Optional

import torch
from random import randint, sample
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from transformers import BertTokenizer


class HypoDataset(IterableDataset):
    def __init__(self,
                 tokenizer: BertTokenizer,
                 corpus_path: Union[str, Path],
                 hypo_index_path: Union[str, Path],
                 train_set_path: Union[str, Path],
                 hypernym_list: List[str],
                 debug: bool = False):
        self.tokenizer = tokenizer
        self.corpus = self._read_corpus(corpus_path)
        self.hypo_index = self._read_json(hypo_index_path)
        self.train_set = self._read_json(train_set_path)
        self.hypernym_to_idx = {hype: n for n, hype in enumerate(hypernym_list)}
        self.debug = debug

    @classmethod
    def _read_json(cls, hypo_index_path: Union[str, Path]):
        with open(hypo_index_path, encoding='utf8') as handle:
            return json.load(handle)

    @staticmethod
    def _read_corpus(corpus_path: Union[str, Path]):
        with open(corpus_path, encoding='utf8') as handle:
            return handle.readlines()

    def __iter__(self):
        while True:
            train_ind = randint(0, len(self.train_set) - 1)
            hypos, hypes, hype_hypes = self.train_set[train_ind]
            hypos_in_index = [h for h in hypos if h in self.hypo_index]

            if not hypos_in_index:
                if self.debug:
                    print(f'Empty index for hypos: {hypos}')
                continue
            if self.debug and len(hypos) != len(hypos_in_index):
                print(f'Some hypos are lost. Original: {hypos},'
                      f' In index: {hypos_in_index}')

            hypo = sample(hypos_in_index, 1)[0]
            all_hypes = list(chain(*(hypes + hype_hypes)))
            hype = sample(all_hypes, 1)[0]
            hype_idx = self.hypernym_to_idx[hype]
            sent_idx, in_sent_hypo_idx = sample(self.hypo_index[hypo], 1)[0]
            sent_toks = self.corpus[sent_idx].split()
            sent_toks = ['[CLS]'] + sent_toks + ['[SEP]']
            subword_idxs, hypo_mask = self._get_indices_and_masks(sent_toks, in_sent_hypo_idx)
            yield subword_idxs, hypo_mask, hype_idx

    def _get_indices_and_masks(self, sent_tokens: List[str], in_sent_hypo_idx: int):
        sent_subword_idxs = []
        sent_subwords = []
        sent_hypo_mask = []
        for n, tok in enumerate(sent_tokens):
            subtokens = self.tokenizer.tokenize(tok)
            sent_subwords.extend(subtokens)
            subtok_idxs = self.tokenizer.convert_tokens_to_ids(subtokens)
            sent_subword_idxs.extend(subtok_idxs)
            mask_value = float(n == in_sent_hypo_idx)
            sent_hypo_mask.extend([mask_value] * len(subtok_idxs))
        return sent_subword_idxs, sent_hypo_mask

    def get_all_hypo_samples(self,
                             hypo: str) \
                             -> Tuple[List[Union[torch.Tensor, List[int], List[float]]]]:
        hypo_mentions = self.hypo_index[hypo]
        sents_indices, sents_hypo_masks = [], []
        for sent_idx, in_sent_hypo_idx in hypo_mentions:
            sent_toks = self.corpus[sent_idx].split()
            subword_idxs, hypo_mask = self._get_indices_and_masks(sent_toks, in_sent_hypo_idx)
            sents_indices.append(subword_idxs)
            sents_hypo_masks.append(hypo_mask)
        batch_parts = self.torchify_and_pad(sents_indices, sents_hypo_masks)
        sents_indices, sents_hypo_masks, sents_att_masks = batch_parts
        return sents_indices, sents_hypo_masks, sents_att_masks

    @classmethod
    def torchify_and_pad(cls,
                         sents_indices: List[List[int]],
                         sents_masks: List[List[float]],
                         hype_idxs: Optional[List[int]] = None) -> Tuple[torch.Tensor]:
        batch_size = len(sents_indices)
        max_len = max(len(idx) for idx in sents_indices)
        padded_indices = torch.zeros(batch_size, max_len, dtype=torch.long)
        padded_mask = torch.zeros(batch_size, max_len, dtype=torch.float)
        padded_att_mask = torch.zeros(batch_size, max_len, dtype=torch.float)

        for n, (sent_idxs, sent_mask) in enumerate(zip(sents_indices, sents_masks)):
            up_to = len(sent_idxs)
            sent_idxs = torch.tensor(sent_idxs)
            sent_mask = torch.tensor(sent_mask)
            padded_indices[n, :up_to] = sent_idxs
            padded_mask[n, :up_to] = sent_mask
            padded_att_mask[n, :up_to] = 1.0
        if hype_idxs:
            hype_idxs = torch.tensor(hype_idxs)
            return padded_indices, padded_mask, padded_att_mask, hype_idxs
        else:
            return padded_indices, padded_mask, padded_att_mask


def get_hypernyms_list_from_train(train_path: Union[Path, str]) -> List[str]:
    train_set = HypoDataset._read_json(train_path)
    hypernyms_set = set()
    for hypos, hypes, hype_hypes in train_set:
        all_hypes = chain(*(hypes + hype_hypes))
        hypernyms_set.update(all_hypes)
    return sorted(hypernyms_set)


def batch_collate(batch: List[Union[List[float], List[int], int]]) -> List[torch.Tensor]:
    """ Pads batch """
    indices, masks, hype_idxs = list(zip(*batch))
    indices, masks, attention_masks, hype_idxs = HypoDataset.torchify_and_pad(indices, masks, hype_idxs)
    return indices, masks, attention_masks, hype_idxs


if __name__ == '__main__':
    tokenizer_vocab_path = 'sample_data/vocab.txt'
    corpus_path = 'sample_data/tst_corpus.txt'
    hypo_index_path = 'sample_data/tst_index.json'
    train_set_path = 'sample_data/tst_train.json'

    tokenizer = BertTokenizer(tokenizer_vocab_path, do_lower_case=False)
    hype_list = get_hypernyms_list_from_train(train_set_path)
    print(f'Hypernym list: {hype_list}')
    ds = HypoDataset(tokenizer,
                     corpus_path,
                     hypo_index_path,
                     train_set_path,
                     hype_list)

    sentence_indices, sentence_hypo_mask, hype_idx = next(iter(ds))
    print(f'Indices: {sentence_indices}\n'
          f'Hypo Mask: {sentence_hypo_mask}')
    print('Note that dataset samples randomly form all possible samples')

    print('=' * 20)

    dl = DataLoader(ds, batch_size=2, collate_fn=batch_collate)
    idxs_batch, mask_batch, attention_masks_batch, hype_idxs = next(iter(dl))
    print(f'Indices batch: {idxs_batch}\n'
          f'Mask batch: {mask_batch}\n'
          f'Attention mask batch: {attention_masks_batch}\n'
          f'Hype indices batch: {hype_idxs}')

    print('='*20)
    hyponym = 'кот'
    batch = ds.get_all_hypo_samples(hyponym)
    indices, masks, att_masks = batch
    print(f'BERT inputs for all mentions of a hyponym {hyponym}\n'
          f'Indices: {indices}\n'
          f'Mask: {masks}\n'
          f'Attention masks: {att_masks}')

# TODO: add repetion of the same hypo
# TODO: separate class
