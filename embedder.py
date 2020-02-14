#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List

import torch
from transformers import BertConfig, BertTokenizer, BertModel


def get_embedding(words: List[str],
                  emb_mat: torch.Tensor,
                  tokenizer: BertTokenizer,
                  debug: bool = False) -> torch.Tensor:
    # emb_mat: [vocab_size, emb_size]
    # returns: [num_words, emb_size]
    subword_ids, subword_masks = [], []
    max_len = 0
    for w in words:
        subword_toks = tokenizer.tokenize(w)
        subword_ids.append(tokenizer.convert_tokens_to_ids(subword_toks))
        num_subwords = len(subword_ids[-1])
        subword_masks.append([1.] * num_subwords)
        if debug:
            print(f"subword_ids('{w}') = {subword_ids[-1]}")
            print(f'{[tokenizer._convert_id_to_token(s) for s in subword_ids[-1]]}')
        max_len = max_len if max_len > num_subwords else num_subwords
    # subword_ids, subword_masks: [num_words, max_len]
    subword_ids = torch.tensor([sw_list + [-1] * (max_len - len(sw_list))
                                for sw_list in subword_ids])
    subword_masks = torch.tensor([m + [0.] * (max_len - len(m))
                                 for m in subword_masks])
    # subword_sizes: [num_words]
    subword_sizes = torch.sum(subword_masks, 1)
    if debug:
        print(subword_sizes)
    # emb_mat[subword_ids]: [num_words, max_len, emb_size]
    return torch.sum(emb_mat[subword_ids] * subword_masks.unsqueeze(2), axis=1) \
        / subword_sizes.unsqueeze(1)


if __name__ == "__main__":
    model_path = Path('/home/hdd/models/rubert_v2/rubert_cased_L-12_H-768_A-12_v2/')

    config = BertConfig.from_pretrained(model_path / 'bert_config.json')
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
    model = BertModel.from_pretrained(str(model_path / 'ptrubert.pt'), config=config)

    emb_mat = model.embeddings.word_embeddings.weight

    res = get_embedding(['мыть', 'мама'], emb_mat, tokenizer=tokenizer)
    print(res.shape)
    print('norms =', res.norm(dim=1))

    res0 = get_embedding(['том', 'бук', 'Двина'], emb_mat, tokenizer=tokenizer)
    print(res0.shape)
    print('norms =', res0.norm(dim=1))


    res1 = get_embedding(['Литва', 'Литовская Республика', 'ывралыва'], emb_mat,
                         tokenizer=tokenizer)[1]
    print(res1.shape)

    res2 = get_embedding(['Литовская', 'Республика'], emb_mat,
                         tokenizer=tokenizer).sum(dim=0) / 2
    print('norms =', res2.norm(dim=0))
    print(res2.shape)

    print("Difference between embeddings of 'Литовская Республика':", (res1 - res2).norm())
