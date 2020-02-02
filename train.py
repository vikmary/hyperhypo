#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, BertModel
from tensorboardX import SummaryWriter

from dataset import HypoDataset, batch_collate, get_hypernyms_list_from_train
from hybert import HyBert

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    default_model_dir = '/home/hdd/models/rubert_cased_L-12_H-768_A-12_v2/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', '-c', type=str, default='news-sample',
                        help='What data to use. Options: ("wiki", '
                             '"news", "wiki-sample", "news-sample")')
    parser.add_argument('--model-dir', '-m', type=Path, default=default_model_dir,
                        help='Path to the directory with model files')
    parser.add_argument('--model-file', type=Path, default='ptrubert.pt')
    parser.add_argument('--only-train-hypes', action='store_true')
    args = parser.parse_args()

    mode = 'train' if args.only_train_hypes else 'full'
    model_dir = args.model_dir
    
    
    data_path = Path('/home/hdd/data/hypernym/')

    corpus = args.corpus
    if corpus == 'news-sample':
        corpus_file = 'corpus.news_dataset-sample.token.txt'
        index_file = f'index.{mode}.news_dataset-sample.json'
    elif corpus == 'wiki-sample':
        corpus_file = 'corpus.wikipedia-ru-2018-sample.token.txt'
        index_file = 'index.train.wikipedia-ru-2018-sample.json'
    else:
        raise NotImplementedError

    corpus_path = data_path / corpus_file
    index_path = data_path / index_file
    train_path = data_path / 'train.cased.json'
    
    model_weights_path = model_dir / 'ptrubert.pt'
    config_path = model_dir / 'bert_config.json'
    tokenizer_vocab_path = model_dir / 'vocab.txt'
    args.corpus_path = corpus_path
    args.index_path = index_path
    args.train_path = train_path
    args.model_weights_path = model_weights_path
    args.config_path = config_path
    args.tokenizer_vocab_path = tokenizer_vocab_path
    args.model_dir = model_dir
    
    return args


def main():
    args = parse_args()
    tokenizer = BertTokenizer(args.tokenizer_vocab_path, do_lower_case=False)

    hype_list = get_hypernyms_list_from_train(args.train_path)
    ds = HypoDataset(tokenizer,
                     args.corpus_path,
                     args.index_path,
                     args.train_path,
                     hype_list)

    dl = DataLoader(ds, batch_size=16, collate_fn=batch_collate)

    config = BertConfig.from_pretrained(args.config_path)
    bert = BertModel.from_pretrained(args.model_weights_path, config=config)
    tokenizer = BertTokenizer.from_pretrained(args.model_dir, do_lower_case=False)

    model = HyBert(bert, tokenizer, hype_list)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    # TODO: add option for passing model.bert.parameters to train embeddings
    optimizer = torch.optim.Adam(model.bert.encoder.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                  1e-6,
                                                  3e-5,
                                                  step_size_up=10000,
                                                  step_size_down=100000,
                                                  cycle_momentum=False)

    writer = SummaryWriter()


    def to_device(*args):
        return tuple(arg.to(device) for arg in args)


    # TODO: add warmap
    # TODO: add gradient accumulation
    count = 0

    running_loss = 0
    exponential_avg = 0.99
    best_loss = 1e9

    save_every = 5000
    for batch in dl:
        if batch[0].shape[1] > 256:
            continue
        idxs_batch, mask_batch, attention_masks_batch, hype_idxs = to_device(*batch)
        model.zero_grad()
        response = model(idxs_batch, mask_batch, attention_masks_batch)
        loss = criterion(response, hype_idxs)
        loss = torch.mean(loss)
        loss.backward()
        writer.add_scalar('log-loss', loss, count)

        if count % save_every == save_every - 1:
            if running_loss < best_loss:
                torch.save(model.parameters(), 'models/checkpoint.pt')
        scheduler.step()
        count += 1
        optimizer.step()

    writer.close()

if __name__ == '__main__':
    main()
