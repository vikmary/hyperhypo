#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from typing import Union, List
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from transformers import BertModel, BertConfig, BertTokenizer

from embedder import get_word_embeddings


class HyBert(nn.Module):
    def __init__(self,
                 bert: BertModel,
                 tokenizer: BertTokenizer,
                 hypernym_list: Union[str, Path, List[str]]):
        super(HyBert, self).__init__()
        self.bert = bert
        if not isinstance(hypernym_list, list):
            hypernym_list = self._read_hypernym_list(hypernym_list)

        self.tokenizer = tokenizer
        self.hypernym_list = hypernym_list
        embeddings = self.bert.embeddings.word_embeddings.weight.data
        self.hypernym_embeddings = get_word_embeddings(self.hypernym_list,
                                                       embeddings.detach(),
                                                       self.tokenizer)

    @staticmethod
    def _read_hypernym_list(hypernym_list_path: Union[str, Path]) -> List[str]:
        with open(hypernym_list_path) as handle:
            return [line.strip() for line in handle]

    def forward(self, indices_batch: LongTensor, hypo_mask: Tensor) -> Tensor:
        h = self.bert(indices_batch)[0]
        m = torch.tensor(hypo_mask).unsqueeze(2)
        hyponym_representations = torch.mean(h * m, 1)
        hypernym_logits = hyponym_representations @ self.hypernym_embeddings.T
        return hypernym_logits


if __name__ == '__main__':
    model_path = Path('/home/arcady/data/models/rubert_cased_L-12_H-768_A-12_v2/')
    model_weights_path = model_path / 'ptrubert.pt'
    config_path = model_path / 'bert_config.json'

    config = BertConfig.from_pretrained(config_path)
    bert = BertModel.from_pretrained(model_weights_path, config=config)
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)

    # Get candidate hypernyms
    with open('sample_data/tst_train.json') as handle:
        train_data = json.load(handle)
    # h - hypernyms, hh - hypernyms of hypernyms
    hypernyms = []
    for _, h, hh in train_data:
        hypernyms.extend(h + hh)

    model = HyBert(bert, tokenizer, hypernyms)

    # example of scoring "кошки" hyponym over the hypernyms presented in training data
    sample_tokes = 'Внимание кошки'
    tokens = tokenizer.tokenize(sample_tokes)
    token_idxs = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
    hyponym_mask = torch.zeros_like(token_idxs, dtype=torch.float)
    hyponym_mask[0, 1] = 1.0
    hypernym_logits = model(token_idxs, hyponym_mask)
    print(f'Subwords: {tokens}\n'
          f'Subword indices [batch_size, seq_len]: {token_idxs}\n'
          f'Hyponym mask [batch_size, seq_len]: {hyponym_mask}\n'
          f'Hypernym logits [batch_size, num_hypernyms]: {hypernym_logits}')
