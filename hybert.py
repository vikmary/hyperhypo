#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from typing import Union, List, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from transformers import BertModel, BertConfig, BertTokenizer

from dataset import get_hypernyms_list_from_train
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
        hype_embeddings = get_word_embeddings(self.hypernym_list,
                                              embeddings.detach(),
                                              self.tokenizer)
        self.hypernym_embeddings = torch.nn.Parameter(hype_embeddings)

    @staticmethod
    def _read_hypernym_list(hypernym_list_path: Union[str, Path]) -> List[str]:
        print(f"Loading candidates from {hypernym_list_path}.")
        with open(hypernym_list_path, 'rt') as handle:
            return [line.strip() for line in handle]

    def forward(self,
                indices_batch: LongTensor,
                hypo_mask: Tensor,
                attention_mask: Tensor) -> Tuple[Tensor, Tensor]:
        # h: [batch_size, seqlen, hidden_size]
        h = self.bert(indices_batch, attention_mask=attention_mask)[0]
        # m: [batch_size, seqlen, 1]
        m = hypo_mask.unsqueeze(2)
        # hyponym_representations: [batch_size, hidden_size]
        hyponym_representations = torch.sum(h * m, 1) / torch.sum(m, 1)
        # hypernym_logits: [batch_size, vocab_size]
        hypernym_logits = hyponym_representations @ self.hypernym_embeddings.T
        hypernym_logits = torch.log_softmax(hypernym_logits, 1)
        return hyponym_representations, hypernym_logits


if __name__ == '__main__':
    model_path = Path('/home/arcady/data/models/rubert_cased_L-12_H-768_A-12_v2/')
    model_weights_path = model_path / 'ptrubert.pt'
    config_path = model_path / 'bert_config.json'

    config = BertConfig.from_pretrained(config_path)
    bert = BertModel.from_pretrained(model_weights_path, config=config)
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)

    # Get candidate hypernyms
    train_set_path = 'sample_data/tst_train.json'
    hypernyms = get_hypernyms_list_from_train(train_set_path)

    model = HyBert(bert, tokenizer, hypernyms)

    # example of scoring "кошки" hyponym over the hypernyms presented in training data
    sample_tokes = 'Внимание кошки'
    tokens = tokenizer.tokenize(sample_tokes)
    token_idxs = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
    hyponym_mask = torch.zeros_like(token_idxs, dtype=torch.float)
    hyponym_mask[0, 1] = 1.0
    _, hypernym_logits = model(token_idxs, hyponym_mask)
    print(f'Subwords: {tokens}\n'
          f'Subword indices [batch_size, seq_len]: {token_idxs}\n'
          f'Hyponym mask [batch_size, seq_len]: {hyponym_mask}\n'
          f'Hypernym logits [batch_size, num_hypernyms]: {hypernym_logits}')
