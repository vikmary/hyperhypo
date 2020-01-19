#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List

import torch
from transformers import BertConfig, BertTokenizer, BertModel


def get_word_embeddings(words: List[str],
                        emb_mat: torch.Tensor,
                        tokenizer: BertTokenizer) -> torch.Tensor:
    # emb_mat: [vocab_size, emb_size]
    # returns: [num_words, emb_size]
    subword_ids, subword_masks = [], []
    max_len = 0
    for w in words:
        subword_ids.append(tokenizer.encode(w, add_special_tokens=False))
        num_subwords = len(subword_ids[-1])
        subword_masks.append([1.] * num_subwords)
        print(f"subword_ids('{w}') = {subword_ids[-1]}")
        print(f'{[tokenizer._convert_id_to_token(s) for s in subword_ids[-1]]}')
        max_len = max_len if max_len > num_subwords else num_subwords
    subword_ids = torch.tensor([sw_list + [-1] * (max_len - len(sw_list))
                                for sw_list in subword_ids])
    subword_masks = torch.tensor([m + [0.] * (max_len - len(m))
                                 for m in subword_masks])
    subword_sizes = torch.sum(subword_masks, 1)
    print(subword_sizes)
    return torch.sum(emb_mat[subword_ids] * subword_masks.unsqueeze(2), axis=1) \
        / subword_sizes.unsqueeze(1)


if __name__ == "__main__":
    model_path = Path('/home/hdd/models/rubert_cased_L-12_H-768_A-12_v2/')

    config = BertConfig.from_pretrained(model_path / 'bert_config.json')
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
    model = BertModel.from_pretrained(str(model_path / 'ptrubert.pt'), config=config)

    emb_mat = model.embeddings.word_embeddings.weight

    res = get_word_embeddings(['мыть', 'мама'], emb_mat, tokenizer=tokenizer)
    print(res, res.shape)
    res = get_word_embeddings(['Литва', 'Литовская Республика', 'ывралыва'], emb_mat, tokenizer=tokenizer)
    print(res, res.shape)
