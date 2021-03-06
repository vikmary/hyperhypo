#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import argparse
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, BertModel
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler

from dataset import batch_collate
from dataset_synonym import SynoDataset, get_synonyms_list_from_train
from hybert import HyBert
from embedder import device, to_device


def parse_args():
    default_model_dir = '/home/hdd/models/rubert_v2/rubert_cased_L-12_H-768_A-12_v2'

    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', '-c', type=str, default='news-sample',
                        choices=('wiki', 'news', 'wiki-sample', 'news-sample'),
                        help='What data to use. Options: ("wiki", '
                             '"news", "wiki-sample", "news-sample")')
    parser.add_argument('--model-dir', '-m', type=Path, default=default_model_dir,
                        help='Path to the directory with model files')
    parser.add_argument('--model-file', type=Path, default='ptrubert.pt')
    parser.add_argument('--trainable-embeddings', action='store_true')
    parser.add_argument('--model-name', type=str, default='checkpoint',
                        help='The name to save the model')
    parser.add_argument('--synset-level', action='store_true')
    parser.add_argument('--predict-one-syno', action='store_true',
                        help='whether predict one synonym in loss or all')
    parser.add_argument('--use-projection', action='store_true',
                        help='learn projection output layer')
    parser.add_argument('--freeze-bert', action='store_true',
                        help='do not update bert parameters')
    parser.add_argument('--lr', default=2e-5, type=float,
                        help='learning rate for training')
    parser.add_argument('--embed-with-special-tokens', action='store_true',
                        help='use [CLS] and [SEP] when embedding phrases')
    args = parser.parse_args()

    mode = 'full'
    model_dir = args.model_dir

    data_path = Path('/home/hdd/data/hypernym/')

    corpus = args.corpus
    if corpus == 'news-sample':
        corpus_file = 'corpus.news_dataset-sample.token.txt'
        index_file = f'index.{mode}.news_dataset-sample.json'
    elif corpus == 'news':
        corpus_file = 'corpus.news_dataset.token.txt'
        index_file = f'index.{mode}.news_dataset.json'
    elif corpus == 'wiki-sample':
        corpus_file = 'corpus.wikipedia-ru-2018-sample.token.txt'
        index_file = f'index.{mode}.wikipedia-ru-2018-sample.json'
    else:
        raise NotImplementedError

    corpus_path = data_path / corpus_file
    index_path = data_path / index_file
    train_path = data_path / 'train_synonyms.cased.not_lemma.json'

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

    level = 'synset' if args.synset_level else 'sense'
    output_synonym_list = get_synonyms_list_from_train(args.train_path, level=level)

    config = BertConfig.from_pretrained(args.config_path)
    bert = BertModel.from_pretrained(args.model_weights_path, config=config)
    tokenizer = BertTokenizer.from_pretrained(args.model_dir, do_lower_case=False)

    model = HyBert(bert,
                   tokenizer,
                   output_synonym_list,
                   use_projection=args.use_projection,
                   embed_with_encoder_output=True,
                   embed_wo_special_tokens=not args.embed_with_special_tokens)

    # initialization = 'models/100k_4.25.pt'
    # print(f'Initializing model from {initialization}')
    # model.load_state_dict(torch.load(initialization))
    model.to(device)
    ds = SynoDataset(tokenizer,
                     args.corpus_path,
                     args.index_path,
                     args.train_path,
                     output_synonym_list,
                     predict_one=not args.predict_one_syno,
                     embed_with_special_tokens=args.embed_with_special_tokens,
                     level=level)

    print(f'Train set len: {len(ds.get_train_idxs())}')
    print(f'Valid set len: {len(ds.get_valid_idxs())}')
    train_sampler = SubsetRandomSampler(ds.get_train_idxs())
    valid_sampler = SubsetRandomSampler(ds.get_valid_idxs())

    # TODO: add optional batch size
    dl_train = DataLoader(ds, batch_size=32, collate_fn=batch_collate,
                          sampler=train_sampler)
    dl_valid = DataLoader(ds, batch_size=32, collate_fn=batch_collate,
                          sampler=valid_sampler)

    criterion = torch.nn.KLDivLoss(reduction='none')
    params = []
    if not args.freeze_bert:
        if args.trainable_embeddings:
            params = list(model.bert.parameters())
        else:
            params = list(model.bert.encoder.parameters())
    if args.use_projection:
        params.extend(model.projection.parameters())
    print(f"Using learning rate equal to {args.lr}.")
    optimizer = torch.optim.Adam(params, lr=args.lr)
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

    epochs = 1000
    save_every = 5000
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        for batch in tqdm(dl_train):
            idxs_batch, mask_batch, attention_masks_batch, syno_idxs = to_device(*batch)
            model.zero_grad()
            _, response = model(idxs_batch, mask_batch, attention_masks_batch)
            loss = criterion(response, syno_idxs)
            loss = torch.sum(loss) / len(idxs_batch)
            loss.backward()
            loss = loss.detach().cpu().numpy()
            if running_loss is None:
                running_loss = loss
            else:
                running_loss = running_loss * exponential_avg + loss * (1 - exponential_avg)
            writer.add_scalar('log-loss', loss, count)

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
                    idxs_batch, mask_batch, attention_masks_batch, syno_idxs = to_device(
                        *batch)
                    model.zero_grad()
                    _, response = model(idxs_batch, mask_batch, attention_masks_batch)
                    loss = criterion(response, syno_idxs)
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
