# Session 7 -Sentiment Analysis

## Objective

- MyCode  [class](https://colab.research.google.com/drive/1zge6bHjZFB-2Sa7MZy4GGqmiJ-5n66GH#scrollTo=ZiWuwQyD8SGt)

ONLY use datasetSentences.txt. (no augmentation required)
Your dataset must have around 12k examples.
Split Dataset into 70/30 Train and Test (no validation)
Convert floating-point labels into 5 classes (0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0) 


## Solution

  N_EPOCHS = 10
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
     
    # train the model
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    
    # evaluate the model
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
    
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% \n')
    
 #-------10 Outputs-----------------------------------------   
Tweet                       prediction           Actual
The verdict : Two bodies and hardly a laugh between them .      very Negative         very Negative
"
The first mistake     Negative         Negative
"
Any movie that makes hard work seem heroic deserves a look .      very Negative         very Negative
"
But is that knot from dramatic tension or a symptom of artistic malnutrition ?      very Negative         very Negative
"
What starts off as a potentially incredibly twisting mystery becomes simply a monster chase film .      very Negative         very Negative
"
Spectacularly beautiful     Neutral         Neutral
"
With very little to add beyond the dark visions already relayed by superb recent predecessors like Swimming With Sharks and The Player     Neutral         Neutral
"
Its direction     Neutral         Neutral
"
The movie is as padded as Allen 's jelly belly .      very Negative         very Negative
"
The trashy teen - sleaze equivalent of Showgirls .      very Negative         very Negative
