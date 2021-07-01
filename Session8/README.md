# Session 8 Torch Text and Advanced Concepts

## Objective

1.Refactor [this](https://github.com/bentrevett/pytorch-seq2seq) repo, change the 2 and 3 (optional 4) such that
- is uses none of the legacy stuff
- It MUST use Multi30k dataset from torchtext
- uses yield_token, and other code that we wrote

## Solution

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/pankaj90382/END-1.0/blob/main/S8/2%20-%20Learning%20Phrase%20Representations%20using%20RNN%20Encoder-Decoder%20for%20Statistical%20Machine%20Translation.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/pankaj90382/END-1.0/blob/main/S8/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb)

### Refactoring

The existing code has written in the torchtext legacy set. The code has refactored to use the latest features from the torchtext version 0.10.0. 

- Refactored Code with torch DataLoader
   ```python
   from torchtext.datasets import Multi30k
  # from torchtext.legacy.data import Field, BucketIterator
  from torchtext.data.utils import get_tokenizer
  from torchtext.vocab import build_vocab_from_iterator
  
   SRC_LANGUAGE = 'de'
   TGT_LANGUAGE = 'en'

  # Place-holders
  token_transform = {}
  vocab_transform = {}
  
  # Create source and target language tokenizer. Make sure to install the dependencies.

  token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de')
  token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en')
  
    # helper function to yield list of tokens
  from typing import Iterable, List
  def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
      language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

      for data_sample in data_iter:
          yield token_transform[language](data_sample[language_index[language]])
          
  # Define special symbols and indices
  UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
  # Make sure the tokens are in order of their indices to properly insert them in vocab
  special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
  
  for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  # Training data Iterator 
  train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
  # Create torchtext's Vocab object 
  vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=2,
                                                    specials=special_symbols,
                                                    special_first=True)
                                                    
   # Set UNK_IDX as the default index. This index is returned when the token is not found. 
   # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary. 
   for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
     vocab_transform[ln].set_default_index(UNK_IDX)
     
  ######################################################################
  # Collation
  # ---------
  #   
  # As seen in the ``Data Sourcing and Processing`` section, our data iterator yields a pair of raw strings. 
  # We need to convert these string pairs into the batched tensors that can be processed by our ``Seq2Seq`` network 
  # defined previously. Below we define our collate function that convert batch of raw strings into batch tensors that
  # can be fed directly into our model.   
  #


  from torch.nn.utils.rnn import pad_sequence

  # helper function to club together sequential operations
  def sequential_transforms(*transforms):
      def func(txt_input):
          for transform in transforms:
              txt_input = transform(txt_input)
          return txt_input
      return func

  # function to add BOS/EOS and create tensor for input sequence indices
  def tensor_transform(token_ids: List[int]):
      return torch.cat((torch.tensor([BOS_IDX]), 
                        torch.tensor(token_ids), 
                        torch.tensor([EOS_IDX])))

  # src and tgt language text transforms to convert raw strings into tensors indices
  text_transform = {}
  for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
      text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                                 vocab_transform[ln], #Numericalization
                                                 tensor_transform) # Add BOS/EOS and create tensor


  # function to collate data samples into batch tesors
  def collate_fn(batch):
      src_batch, tgt_batch = [], []
      for src_sample, tgt_sample in batch:
          src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
          tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

      src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
      tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
      return src_batch, tgt_batch
      
   BATCH_SIZE = 128

  from torchtext.data.functional import to_map_style_dataset
  from torch.utils.data import DataLoader
  train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
  train_dataloader = DataLoader(to_map_style_dataset(train_iter), shuffle=True, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, collate_fn=collate_fn)

  val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
  val_dataloader = DataLoader(to_map_style_dataset(val_iter), shuffle=True, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, collate_fn=collate_fn)

  test_iter = Multi30k(split='test', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
  test_dataloader = DataLoader(to_map_style_dataset(test_iter), shuffle=True, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, collate_fn=collate_fn)
   ```

### Logs

```
  Epoch: 01 | Time: 0m 37s
   Train Loss: 5.042 | Train PPL: 154.706
    Val. Loss: 4.995 |  Val. PPL: 147.617
  Epoch: 02 | Time: 0m 38s
   Train Loss: 4.370 | Train PPL:  79.051
    Val. Loss: 4.794 |  Val. PPL: 120.789
  Epoch: 03 | Time: 0m 38s
   Train Loss: 4.042 | Train PPL:  56.935
    Val. Loss: 4.524 |  Val. PPL:  92.220
  Epoch: 04 | Time: 0m 39s
   Train Loss: 3.715 | Train PPL:  41.069
    Val. Loss: 4.313 |  Val. PPL:  74.630
  Epoch: 05 | Time: 0m 38s
   Train Loss: 3.415 | Train PPL:  30.430
    Val. Loss: 4.085 |  Val. PPL:  59.464
  Epoch: 06 | Time: 0m 39s
   Train Loss: 3.136 | Train PPL:  23.013
    Val. Loss: 3.891 |  Val. PPL:  48.973
  Epoch: 07 | Time: 0m 39s
   Train Loss: 2.873 | Train PPL:  17.688
    Val. Loss: 3.810 |  Val. PPL:  45.130
  Epoch: 08 | Time: 0m 39s
   Train Loss: 2.629 | Train PPL:  13.855
    Val. Loss: 3.726 |  Val. PPL:  41.526
  Epoch: 09 | Time: 0m 39s
   Train Loss: 2.399 | Train PPL:  11.017
    Val. Loss: 3.643 |  Val. PPL:  38.194
  Epoch: 10 | Time: 0m 39s
   Train Loss: 2.177 | Train PPL:   8.820
    Val. Loss: 3.705 |  Val. PPL:  40.649
```

### Concepts


| Colab Files| Modern | Class Notes | Torch Text Legacy|
|-------|-----|-----|-----|
| Seq2Seq   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/pankaj90382/END-1.0/blob/main/S8/END2.0%20Seq2Seq%201%20Modern.ipynb)  |   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/pankaj90382/END-1.0/blob/main/S8/torchtext%20legacy%20END2%20Seq2seq%20Class%20Code.ipynb) |
| AGT News Classification  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/pankaj90382/END-1.0/blob/main/S8/torchtext%20AGT_News_Exercises.ipynb) |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/pankaj90382/END-1.0/blob/main/S8/AGT_News_Classification_ClassLive.ipynb)|  |


## Refrences

 - [Loss Function and Optimization](https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c)
 - [Torch Text Datasets](https://pytorch.org/text/stable/datasets.html#ag-news)
 - [Pytorch Seq2seq Dataset](https://github.com/bentrevett/pytorch-seq2seq)
 - [Pytorch Language Translation Tutorial](https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html)
 - [Pytorch Sentiment Tutorial](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)
 - [Pytorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
 - [Pytorch Tutorials](https://pytorch.org/tutorials/)
