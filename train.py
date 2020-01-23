from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, BertModel

from dataset import HypoDataset, batch_collate, get_hypernyms_list_from_train
from hybert import HyBert

tokenizer_vocab_path = 'sample_data/vocab.txt'
corpus_path = 'sample_data/tst_corpus.txt'
hypo_index_path = 'sample_data/tst_index.json'
train_set_path = 'sample_data/tst_train.json'

model_path = Path('/home/arcady/data/models/rubert_cased_L-12_H-768_A-12_v2/')
model_weights_path = model_path / 'ptrubert.pt'
config_path = model_path / 'bert_config.json'

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
criterion = torch.nn.CrossEntropyLoss()
# TODO: add option for passing model.bert.parameters to train embeddings
optimizer = torch.optim.Adam(model.bert.encoder.parameters(), lr=1e-5)

for idxs_batch, mask_batch, attention_masks_batch, hype_idxs in dl:
    response = model(idxs_batch, mask_batch, attention_masks_batch)
    loss = criterion(response, hype_idxs)
    print(loss)
    optimizer.step()
    break