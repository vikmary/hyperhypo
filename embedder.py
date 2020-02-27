#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List

import torch
from dataset import HypoDataset
from transformers import BertConfig, BertTokenizer, BertModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_device(*args):
    return tuple(arg.to(device) for arg in args)


def get_encoder_embedding(phrases: List[str],
                          bert: BertModel,
                          tokenizer: BertTokenizer,
                          embed_wo_special_tokens: bool) -> torch.Tensor:
    subtok_ids_list, hypo_mask_list = [], []
    for phr in phrases:
        subtok_ids_list.append(tokenizer.convert_tokens_to_ids(
            ['[CLS]'] + tokenizer.tokenize(phr) + ['[SEP]']
        ))
        hypo_mask_list.append([1.0] * len(subtok_ids_list[-1]))
        if embed_wo_special_tokens:
            hypo_mask_list[-1][0] = 0.0
            hypo_mask_list[-1][-1] = 0.0
    batch = HypoDataset.torchify_and_pad(subtok_ids_list, hypo_mask_list)
    subtok_ids_batch, hypo_mask_batch, attn_mask_batch = to_device(*batch)
    h = bert(subtok_ids_batch, attention_mask=attn_mask_batch)[0]
    m = hypo_mask_batch.unsqueeze(2)
    phrase_representations = torch.sum(h * m, 1) / torch.sum(m, 1)
    return phrase_representations


def get_embedding(phrases: List[str],
                  emb_mat: torch.Tensor,
                  tokenizer: BertTokenizer,
                  debug: bool = False) -> torch.Tensor:
    # emb_mat: [vocab_size, emb_size]
    # returns: [num_phrases, emb_size]
    subtok_ids, subtok_masks = [], []
    max_len = 0
    for w in phrases:
        subtok_toks = tokenizer.tokenize(w)
        subtok_ids.append(tokenizer.convert_tokens_to_ids(subtok_toks))
        num_subtoks = len(subtok_ids[-1])
        subtok_masks.append([1.] * num_subtoks)
        if debug:
            print(f"subtok_ids('{w}') = {subtok_ids[-1]}")
            print(f'{[tokenizer._convert_id_to_token(s) for s in subtok_ids[-1]]}')
        max_len = max_len if max_len > num_subtoks else num_subtoks
    # subtok_ids, subtok_masks: [num_phrases, max_len]
    subtok_ids = torch.tensor([sw_list + [-1] * (max_len - len(sw_list))
                               for sw_list in subtok_ids])
    subtok_masks = torch.tensor([m + [0.] * (max_len - len(m))
                                 for m in subtok_masks])
    # subtok_sizes: [num_phrases]
    subtok_sizes = torch.sum(subtok_masks, 1)
    if debug:
        print(subtok_sizes)
    # emb_mat[subtok_ids]: [num_phrases, max_len, emb_size]
    return torch.sum(emb_mat[subtok_ids] * subtok_masks.unsqueeze(2), axis=1) \
        / subtok_sizes.unsqueeze(1)


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
