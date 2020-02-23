# Prediction based on definitions

Input:
  - hyponym phrase
    - ЯЧМЕНЬ 
  - ranked list of phrase definitions (from the most used to the most rare)
    1. растение семейства мятликовых, хлебный злак 
    2. зерна ячменя как пищевой продукт 
    3. гнойное воспаление волосяного мешочка ресницы или сальной железы около луковицы ресницы
  - hypernym synsets
    - 
Output:
  - ranked hypernym synsets: (ЗЛАКОВАЯ КУЛЬТУРА, РАСТЕНИЕ), (ЗЕРНО,), (БОЛЕЗНЬ)...

### 0. Encoding senses (phrases)

(0.0) Encode phrase with input BERT embeddins (a bug there?)
  - хирургическая операция -> BertModel.word_embeddings -> average over subtokens

(0.1) Encode phrase with BERT encoder and average (current approach)
  - [CLS] хирургическая операция [SEP] -> BERT -> average over subtokens

(0.2) Encode phrase with BERT encoder (current approach) and take emb_[CLS] 
  - [CLS] хирургическая операция [SEP] -> BERT -> take [CLS]-th representation 
// Take SentBert for the task?

(0.3) Encode phrase's definition with BERT encoder and average over all subtokens
  - [CLS] растение семейства мятликовых, хлебный злак [SEP] -> BERT ->
    average over subtokens

(0.4) Encode phrase's definition with BERT encoder and take emb_[CLS]
  - [CLS] растение семейства мятликовых, хлебный злак [SEP] -> BERT ->
    take [CLS]-th representation

(0.5) Take phrase and it's definition separated by [SEP] (or hyphen) and
average over all subtokens
  - [CLS] ячмень [SEP] растение семейства мятликовых, хлебный злак [SEP] -> BERT ->
    average over subtokens

(0.6) Take phrase and it's definition separated by [SEP] and take emb_[CLS]
  - [CLS] ячмень [SEP] растение семейства мятликовых, хлебный злак [SEP] -> BERT ->
    take [CLS]-th representation

(0.7) Take phrase and it's definition separated by [SEP] and average subtokens for phrase
  - [CLS] ячмень [SEP] растение семейства мятликовых, хлебный злак [SEP] -> BERT ->
    average subtokens for 'ячмень'

### 1. Encoding synsets:

(1.0) Encode one random phrase in synset with (0.0)-(0.7)

(1.1) Encode each phrase in synset with (0.0)-(0.7) and average all

### 2. Train

(2.0) Cross-entropy loss over hypernyms for each hyponym

  (2.0.0) Precalculate embeddings for hypernyms and freeze them

  (2.0.1) Update hypernym embeddings during training
    - sample negatives from the same batch

(2.1) Pair-wise classification for each hypernym, hypernym
