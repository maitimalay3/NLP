# Session 8 Torch Text and Advanced Concepts

## Objective
1. Perform 1 full feed forward step for the encoder manually
2. Perform 1 full feed forward step for the decoder manually.
3. Use the attention mechanism which is combination of Bahdanau and Luong

## Solution

[![Open In Colab](https://colab.research.google.com/drive/15iB4hT4X04kQ4ZX3CfmaYPWzqNWmoAnj#scrollTo=EkPpRywbRxvd)


### Feed Froward Steps

1. Break each sentence to words
2. Find index for each words
3. Get the word embedding vector or word representative vector for each words and get into a tensor shape
4. Intiate LSTM object
5. intitate encoder hidden and cell values values with zeros
6. initiate encoder output with zeros
7. Use the embedding vector of first word and the intial hidden and cell state value, feed it to LSTM cell and get the first output embedding vctor, first hidden output state and first 
8. Continue all the above steps till we reach the last word

### Decoder Steps
1. First input to the decoder will be SOS_token, later inputs would be the words it predicted 
2.  decoder/LSTM's hidden and cell state will be initialized with the encoder's last hidden state and cell state 
3.  Used LSTM's hidden state, cell state and last prediction to generate attention weight using a FC layer. this attention weight will be used to weigh the encoder_outputs using batch matric multiplication. This will give us a NEW view on how to look at encoder_states. 
4.  this attention applied encoder_states will then be concatenated with the input, and then sent a linear layer and softmax layer to normilize and then sent to the LSTM. 
5.  LSTM's output will be sent to a FC layer to predict one of the output_language words 

### Attention Mechanism
 ## For first time 
1. concat these three tensor 1. Decoder embedding word (SOS token), 2. hidden state (last hidden state of encoder) 3. cell state (last cell state of encoder)
2. Use the above conatinated data set and pass it though a fully connected layer and then normalize it using softmax to to get the normalized attention weights
3. This attention weight will be used to weigh the encoder_outputs using batch matric multiplication and get new representation of encoder state
4. Concatinate the new attention weighted encoder state (build using step 3) with  embedding of the input word to create the input vector for decoder LSTM cell
5. 
