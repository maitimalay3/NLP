# Session 8 Torch Text and Advanced Concepts

## Objective

1.Refactor [this](https://github.com/bentrevett/pytorch-seq2seq) repo, change the 2 and 3 (optional 4) such that
- is uses none of the legacy stuff
- It MUST use Multi30k dataset from torchtext
- uses yield_token, and other code that we wrote

## Solution

[![Open In Colab](https://drive.google.com/file/d/1Dnu0RqhFLYO8fW_ndVZI7IHXOinQ8N2F/view?usp=sharing)
[![Open In Colab](https://drive.google.com/file/d/1codVuImy0ZGQ8SJA_JYasvTvW-gGBDbW/view?usp=sharing)

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



