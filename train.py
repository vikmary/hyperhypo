#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, BertModel
from tensorboardX import SummaryWriter

from dataset import HypoDataset, batch_collate, get_hypernyms_list_from_train
from hybert import HyBert

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# tokenizer_vocab_path = 'sample_data/vocab.txt'
# corpus_path = 'sample_data/tst_corpus.txt'
# hypo_index_path = 'sample_data/tst_index.json'
# train_set_path = 'sample_data/tst_train.json'

data_path = Path('/home/hdd/data/hypernym/')
# corpus_path = data_path / 'corpus.news_df-sample.token.txt'
corpus_path = data_path / 'corpus.wikipedia-ru-2018-sample.token.txt'
# hypo_index_path = data_path / 'index.train.news_df-sample.json'
hypo_index_path = data_path / 'index.train.wikipedia-ru-2018-sample.json'
train_set_path = data_path / 'train.cased.json'
model_path = Path('/home/hdd/models/rubert_cased_L-12_H-768_A-12_v2/')
model_weights_path = model_path / 'ptrubert.pt'
config_path = model_path / 'bert_config.json'
tokenizer_vocab_path = model_path / 'vocab.txt'

tokenizer = BertTokenizer(tokenizer_vocab_path, do_lower_case=False)

hype_list = get_hypernyms_list_from_train(train_set_path)
ds = HypoDataset(tokenizer,
                 corpus_path,
                 hypo_index_path,
                 train_set_path,
                 hype_list)

dl = DataLoader(ds, batch_size=2, collate_fn=batch_collate)

config = BertConfig.from_pretrained(config_path)
bert = BertModel.from_pretrained(model_weights_path, config=config)
tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)

model = HyBert(bert, tokenizer, hype_list)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
# TODO: add option for passing model.bert.parameters to train embeddings
optimizer = torch.optim.Adam(model.bert.encoder.parameters(), lr=3e-5)
writer = SummaryWriter()

def to_device(*args):
    return tuple(arg.to(device) for arg in args)

# TODO: add warmap
# TODO: add gradient accumulation
for batch in dl:
    if batch[0].shape[1] > 256:
        continue
    idxs_batch, mask_batch, attention_masks_batch, hype_idxs = to_device(*batch)
    model.zero_grad()
    response = model(idxs_batch, mask_batch, attention_masks_batch)
    loss = criterion(response, hype_idxs)
    loss.backward()
    writer.add_scalar('log-loss', loss)
    optimizer.step()

writer.close()
