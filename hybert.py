#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from typing import Union, List, Tuple, Dict
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from transformers import BertModel, BertConfig, BertTokenizer

from dataset import get_hypernyms_list_from_train, HypoDataset
from embedder import get_embedding, get_encoder_embedding, to_device, device


class HyBert(nn.Module):
    def __init__(self,
                 bert: BertModel,
                 tokenizer: BertTokenizer,
                 hypernym_list: Union[str, Path, List[List[str]]],
                 embed_with_encoder_output: bool = True,
                 mask_special_tokens: bool = True,
                 use_projection: bool = False,
                 batch_size: int = 128):
        super(HyBert, self).__init__()

        self.bert = bert.to(device)
        if not isinstance(hypernym_list, (list, dict)):
            hypernym_list = self._read_hypernym_list(hypernym_list)

        self.tokenizer = tokenizer
        self.hypernym_list = hypernym_list
        self.use_projection = use_projection

        print(f"Building matrix of hypernym embeddings.")
        self.hypernym_embeddings = \
            torch.nn.Parameter(self._build_hypernym_matrix(hypernym_list,
                                                           embed_with_encoder_output,
                                                           mask_special_tokens,
                                                           batch_size))
        if self.use_projection:
            self.projection = nn.Linear(768, 768)

    @staticmethod
    def _read_hypernym_list(hypernym_list_path: Union[str, Path]) -> List[str]:
        print(f"Loading candidates from {hypernym_list_path}.")
        with open(hypernym_list_path, 'rt') as handle:
            return [line.strip() for line in handle]

    def _build_hypernym_matrix(self,
                               hypernym_list: List[List[str]],
                               embed_with_encoder_output: bool,
                               mask_special_tokens: bool,
                               batch_size: int) -> torch.Tensor:
        if not embed_with_encoder_output:
            embeddings = self.bert.embeddings.word_embeddings.weight.data.detach()
            embeddings = to_device(*[embeddings])

        hype_embeddings = []
        phr_batch, hype_indices = [], []
        for h_id, phrases in tqdm(enumerate(hypernym_list), total=len(hypernym_list)):
            hype_indices.append([len(phr_batch), len(phrases)])
            phr_batch.extend(phrases)
            if (len(phr_batch) >= batch_size) or (h_id == len(hypernym_list) - 1):
                if embed_with_encoder_output:
                    phr_embs_batch = \
                        get_encoder_embedding(phr_batch, self.bert, self.tokenizer,
                                              mask_special_tokens=mask_special_tokens)
                else:
                    phr_embs_batch = get_embedding(phr_batch, embeddings, self.tokenizer)
                phr_embs_batch = phr_embs_batch.detach()
                for h_start, h_num_phrs in hype_indices:
                    h_phr_embs = phr_embs_batch[h_start: h_start + h_num_phrs]
                    hype_embeddings.append(h_phr_embs.mean(dim=0))
                phr_batch, hype_indices = [], []
        return torch.stack(hype_embeddings, dim=0)

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
        if self.use_projection:
            hyponym_representations = self.projection(hyponym_representations)
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
