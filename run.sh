#!/bin/sh

# TODO: add downloading of udpipe model
git clone git@github.com:ufal/udpipe.git
cd udpipe/src
make
cd ../../
# TODO: add code to download and convertion of bert

python3 -m prepare_corpora.preprocess_corpora --data-path corpora/wikipedia-ru-2018.txt -t 4
python3 -m prepare_corpora.preprocess_corpora --data-path corpora/news_dataset.zip -t 4 --news

udpipe/src/udpipe --tokenize --tag data/russian-syntagrus-ud-2.4-190531.udpipe < corpora/corpus.news_dataset.token.txt

# ./train_valid_split.py --data-path taxonomy-enrichment/data/training_data/training.verbs --wordnet-dir taxonomy-enrichment/data/ --valid-rate 0.2
# ./train_valid_split.py --data-path taxonomy-enrichment/data/training_data/training.nouns --wordnet-dir taxonomy-enrichment/data/ --valid-rate 0.1

# python3 taxonomy-enrichment-copy/data/convert_train_to_reference.py taxonomy-enrichment-copy/data/training_data/training_nouns.valid.tsv taxonomy-enrichment-copy/data/training_data/training_nouns.valid.reference.tsv
# python3 taxonomy-enrichment-copy/data/convert_train_to_reference.py taxonomy-enrichment-copy/data/training_data/training_verbs.valid.tsv taxonomy-enrichment-copy/data/training_data/training_verbs.valid.reference.tsv

python3 -m prepare_corpora.build_index --data-path corpora/news_dataset.lemma.txt.gz --train-paths taxonomy-enrichment/data/training_data/training_*.tsv --synset-info-paths taxonomy-enrichment/data/training_data/synsets_*.tsv
python3 -m prepare_corpora.build_index --data-path corpora/wikipedia-ru-2018.lemma.txt.gz --train-paths taxonomy-enrichment/data/training_data/training_*.tsv --synset-info-paths taxonomy-enrichment/data/training_data/synsets_*.tsv

./generate_bert_train.py --data-paths taxonomy-enrichment/data/training_data/training*.tsv --synset-info-paths taxonomy-enrichment/data/training_data/synsets_*.tsv --wordnet-dir taxonomy-enrichment/data/ -o corpora/train.cased.json --bert-model-path /home/hdd/models/rubert_v2/rubert_cased_L-12_H-768_A-12_v2
./generate_bert_train.py --data-path taxonomy-enrichment/data/training_data/dev*.tsv --synset-info-paths taxonomy-enrichment/data/training_data/synsets_*.tsv --wordnet-dir taxonomy-enrichment/data/ -o corpora/valid.cased.json --bert-model-path /home/hdd/models/rubert_v2/rubert_cased_L-12_H-768_A-12_v2
./generate_candidates.py -w ./taxonomy-enrichment/data/ -o corpora/candidates.cased.tsv -b /home/hdd/models/rubert_v2/rubert_cased_L-12_H-768_A-12_v2

# ./generate_labels.py -w taxonomy-enrichment/data -d taxonomy-enrichment/data/training_data/training.verbs.valid -o taxonomy-enrichment/data/training_data
# ./generate_labels.py -w taxonomy-enrichment/data -d taxonomy-enrichment/data/training_data/training.nouns.valid -o taxonomy-enrichment/data/training_data

# Predict on valid samples
CUDA_VISIBLE_DEVICES="0" ./generate_predictions.py -d taxonomy-enrichment/data/training_data/dev_nouns.tsv -w taxonomy-enrichment/data -b /home/hdd/models/rubert_v2/rubert_cased_L-12_H-768_A-12_v2 -o preds/ -c ./corpora/candidates.cased.tsv -i corpora/index.full.news_dataset-sample.json --pos nouns
CUDA_VISIBLE_DEVICES="0" ./generate_predictions.py -d taxonomy-enrichment/data/training_data/dev_verbs.tsv -w taxonomy-enrichment/data -b /home/hdd/models/rubert_v2/rubert_cased_L-12_H-768_A-12_v2 -o preds/ -c ./corpora/candidates.cased.tsv -i corpora/index.full.news_dataset-sample.json --pos verbs

# Predict on public test samples
CUDA_VISIBLE_DEVICES="0" ./generate_predictions.py -d taxonomy-enrichment/data/public_test/nouns_public.tsv --pos nouns -w taxonomy-enrichment/data -b /home/hdd/models/rubert_v2/rubert_cased_L-12_H-768_A-12_v2 -o preds/ -c ./corpora/candidates.cased.tsv -t corpora/corpus.news_dataset-sample.token.txt -f preds/regularized_synonym_k1_all.nouns.fixed.tsv
CUDA_VISIBLE_DEVICES="0" ./generate_predictions.py -d taxonomy-enrichment/data/public_test/verbs_public.tsv --pos verbs -w taxonomy-enrichment/data -b /home/hdd/models/rubert_v2/rubert_cased_L-12_H-768_A-12_v2 -o preds/ -c ./corpora/candidates.cased.tsv -t corpora/corpus.news_dataset-sample.token.txt -f preds/regularized_synonym_k1_all.nouns.fixed.tsv

# Predict on private test samples
CUDA_VISIBLE_DEVICES="0" ./generate_predictions.py -d taxonomy-enrichment/data/private_test/nouns_private.tsv --pos nouns -w taxonomy-enrichment/data -b /home/hdd/models/rubert_v2/rubert_cased_L-12_H-768_A-12_v2 -o preds/ -c ./corpora/candidates.cased.tsv -t corpora/corpus.news_dataset-sample.token.txt -f preds/regularized_synonym_k1_all.nouns.fixed.tsv
CUDA_VISIBLE_DEVICES="0" ./generate_predictions.py -d taxonomy-enrichment/data/private_test/verbs_private.tsv --pos verbs -w taxonomy-enrichment/data -b /home/hdd/models/rubert_v2/rubert_cased_L-12_H-768_A-12_v2 -o preds/ -c ./corpora/candidates.cased.tsv -t corpora/corpus.news_dataset-sample.token.txt -f preds/regularized_synonym_k1_all.nouns.fixed.tsv

# generate subsumptions
# ./generate_subsumptions.py -t ./taxonomy-enrichment/data/training_data/training.nouns ./taxonomy-enrichment/data/training_data/training.verbs -o data/subsumptions-dialog.txt -w ./taxonomy-enrichment/data/
