#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
from pathlib import Path
import argparse
# import tracemalloc
import gc

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, BertModel
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler

from dataset import HypoDataset, batch_collate, get_hypernyms_list_from_train
from hybert import HyBert
from embedder import device, to_device


def parse_args():
    # default_model_dir = '/home/hdd/models/rubert_cased_L-12_H-768_A-12_v2/'
    default_model_dir = '/home/hdd/models/rubert_v2/rubert_cased_L-12_H-768_A-12_v2'

    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', '-c', type=str, default='news-sample',
                        help='What data to use. Options: ("wiki", '
                             '"news", "wiki-sample", "news-sample")')
    parser.add_argument('--model-dir', '-m', type=Path, default=default_model_dir,
                        help='Path to the directory with model files')
    parser.add_argument('--model-file', type=Path, default='ptrubert.pt')
    parser.add_argument('--only-train-hypes', action='store_true')
    parser.add_argument('--trainable-embeddings', action='store_true')
    parser.add_argument('--model-name', type=str, default='checkpoint',
                        help='The name to save the model')
    parser.add_argument('--synset-level', action='store_true')
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
    valid_path = data_path / 'valid.cased.json'
    
    model_weights_path = model_dir / 'ptrubert.pt'
    config_path = model_dir / 'bert_config.json'
    tokenizer_vocab_path = model_dir / 'vocab.txt'
    args.corpus_path = corpus_path
    args.index_path = index_path
    args.train_path = train_path
    args.valid_path = valid_path
    args.model_weights_path = model_weights_path
    args.config_path = config_path
    args.tokenizer_vocab_path = tokenizer_vocab_path
    args.model_dir = model_dir
    
    return args


def main():

    args = parse_args()
    level = 'synset' if args.synset_level else 'sense'
    hype_list = get_hypernyms_list_from_train(args.train_path, level)
    # print(hype_list)
    # raise RuntimeError
    valid_hype_list = get_hypernyms_list_from_train(args.valid_path, level)
    hype_list.extend(valid_hype_list)

    config = BertConfig.from_pretrained(args.config_path)
    bert = BertModel.from_pretrained(args.model_weights_path, config=config)
    tokenizer = BertTokenizer.from_pretrained(args.model_dir, do_lower_case=False)

    model = HyBert(bert, tokenizer, hype_list)

    # initialization = 'models/100k_4.25.pt'
    # print(f'Initializing model from {initialization}')
    # model.load_state_dict(torch.load(initialization))
    model.to(device)
    # tracemalloc.start()
    ds = HypoDataset(tokenizer,
                     args.corpus_path,
                     args.index_path,
                     args.train_path,
                     hype_list,
                     valid_set_path=args.valid_path,
                     level=level)

    print(f'Train set len: {len(ds.get_train_idxs())}')
    print(f'Valid set len: {len(ds.get_valid_idxs())}')
    train_sampler = SubsetRandomSampler(ds.get_train_idxs())
    valid_sampler = SubsetRandomSampler(ds.get_valid_idxs())

    # TODO: add optional batch size
    dl_train = DataLoader(ds, batch_size=44, collate_fn=batch_collate,
                          sampler=train_sampler)
    dl_valid = DataLoader(ds, batch_size=44, collate_fn=batch_collate,
                          sampler=valid_sampler)

    criterion = torch.nn.KLDivLoss(reduction='none')
    # TODO: add option for passing model.bert.parameters to train embeddings
    if args.trainable_embeddings:
        optimizer = torch.optim.Adam(model.bert.parameters(), lr=2e-5)
    else:
        optimizer = torch.optim.Adam(model.bert.encoder.parameters(), lr=2e-5)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
    #                                               1e-5,
    #                                               3e-5,
    #                                               step_size_up=10000,
    #                                               step_size_down=100000,
    #                                               cycle_momentum=False)

    writer = SummaryWriter(f'runs/{args.model_name}')

    # TODO: add warmap
    # TODO: add gradient accumulation
    count = 0

    running_loss = None
    exponential_avg = 0.99
    best_loss = 1e9
    best_val_loss = 1e9

    ecpochs = 1000
    save_every = 5000
    for epoch in range(ecpochs):
        print(f'Epoch: {epoch}')
        for batch in tqdm(dl_train):
            idxs_batch, mask_batch, attention_masks_batch, hype_idxs = to_device(*batch)
            model.zero_grad()
            _, response = model(idxs_batch, mask_batch, attention_masks_batch)
            loss = criterion(response, hype_idxs)
            loss = torch.sum(loss) / len(idxs_batch)
            loss.backward()
            loss = loss.detach().cpu().numpy()
            if running_loss is None:
                running_loss = loss
            else:
                running_loss = running_loss * exponential_avg + loss * (1 - exponential_avg)
            writer.add_scalar('log-loss', loss, count)
            # if count % 100 == 99:
                # snapshot = tracemalloc.take_snapshot()
                #
                # with open(f'memstat/stats_{count}.pckl', 'wb') as fin:
                #     pickle.dump(snapshot, fin)
                # print(f'=================  {count} ======================')
                # print(snapshot.statistics('lineno')[:50])

            # if count % save_every == save_every - 1:
            #     if running_loss < best_loss:
            #         print(f'Better loss achieved {running_loss}! Saving model.')
            #         torch.save(model.state_dict(), 'models/checkpoint.pt')
            # scheduler.step()
            count += 1
            optimizer.step()
            if count % 100 == 99:
                gc.collect()
        # Validation
        val_loss = 0
        val_count = 0
        with torch.no_grad():
            # Average on 10 runs due to sampling of indexed examples
            for _ in tqdm(range(10)):
                for batch in dl_valid:
                    idxs_batch, mask_batch, attention_masks_batch, hype_idxs = to_device(
                        *batch)
                    model.zero_grad()
                    _, response = model(idxs_batch, mask_batch, attention_masks_batch)
                    loss = criterion(response, hype_idxs)
                    loss = torch.sum(loss)
                    val_count += len(idxs_batch)
                    val_loss += loss.detach().cpu().numpy()
        val_loss = val_loss / val_count
        print(f'Validation loss: {val_loss}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f'Better loss achieved {val_loss}! Saving model.')
            # TODO: add model name
            torch.save(model.state_dict(), f'models/{args.model_name}.pt')
        writer.add_scalar('log-loss-val', val_loss, epoch)

    writer.close()


if __name__ == '__main__':
    main()
