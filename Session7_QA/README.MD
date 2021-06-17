# Session 5 - Data Augmentation 

## Objective

- Refer to the file we wrote in the [class](https://colab.research.google.com/drive/1BtOq1qaIezD-_Qbv-yvPjVSq9-YECWq3#scrollTo=GoQGpRxmYhB-)

Use "Back Translate", "random_swap" and "random_delete" to augment the data you are training on
Download the StanfordSentimentAnalysis Dataset from this link  (Links to an external site.)(it might be troubling to download it, so force download on chrome). Use "datasetSentences.txt" and "sentiment_labels.txt" files from the zip you just downloaded as your dataset. This dataset contains just over 10,000 pieces of Stanford data from HTML files of Rotten Tomatoes. 
The sentiments are rated between 0 and 4 , where zero is the most negative and 4 is the most positive.
Train your model and achieve 60%+ validation/text accuracy. Upload your collab file on GitHub with readme that contains details about your assignment/word (minimum 250 words), 
training logs showing final validation accuracy, and outcomes for 10 example inputs from the test/validation data.

## Solution
Implemented  "Back Translate", "random_swap" to improve the accuracy.
validation accuracy improved from 48% to 53% after using "random swap"
However, after using back translate the validation accuracy reduced to 43%, so not used in final solution

Please find below the code related to "Back Translate", "random_swap"

''' Python
def remove_stopwords_swap(example_sent):
  stop_words = set(stopwords.words('english'))
  myset = {".", "-", "...", "s", "'"}

  word_tokens = word_tokenize(example_sent)
  filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
  filtered_sentence = []
  rndpos = randrange(len(word_tokens)-1)
  counter = 0
  
  for w in word_tokens:
    
    if (w not in stop_words) and (w not in myset):
      if counter == rndpos:
        syns = wordnet.synsets(w)
        try:
          if syns[0].lemmas()[0].name() is not None:
            rw = syns[0].lemmas()[0].name()
            #print(w,rw)
            filtered_sentence.append(rw)
          else:
              filtered_sentence.append(w)
        except:
            pass


      else:
        filtered_sentence.append(w)
        #print(counter, rndpos)
        #print(filtered_sentence)
    counter = counter +1   
    
    final = ' '.join(filtered_sentence).split(', ')
    #final = [[' '.join(i)] for i in filtered_sentence]
  
  return final[0]
'''
''' Python
import random
import googletrans
from googletrans import Translator
#import googletrans.Translator
def translation(sentence):
  translator = Translator()
  available_langs = list(googletrans.LANGUAGES.keys()) 
  trans_lang = random.choice(available_langs) 
  #print(f"Translating to {googletrans.LANGUAGES[trans_lang]}")

  
  try:
    translations = translator.translate(sentence, dest=trans_lang)
    t_text = [t.text for t in translations]
    translations_en_random = translator.translate(t_text, src=trans_lang, dest='en') 
    en_text = [t.text for t in translations_en_random]
  except:
    en_text = sentence


  #print(t_text)

  
  return en_text
'''
#### Training Logs

```
Train Loss: 1.375 | Train Acc: 52.89%
	 Val. Loss: 1.393 |  Val. Acc: 52.01% 

	Train Loss: 1.372 | Train Acc: 53.23%
	 Val. Loss: 1.394 |  Val. Acc: 52.01% 

	Train Loss: 1.368 | Train Acc: 53.78%
	 Val. Loss: 1.395 |  Val. Acc: 51.90% 

	Train Loss: 1.364 | Train Acc: 54.62%
	 Val. Loss: 1.396 |  Val. Acc: 51.45% 

	Train Loss: 1.359 | Train Acc: 55.74%
	 Val. Loss: 1.397 |  Val. Acc: 51.06% 

	Train Loss: 1.353 | Train Acc: 56.64%
	 Val. Loss: 1.398 |  Val. Acc: 50.39% 

	Train Loss: 1.345 | Train Acc: 57.56%
	 Val. Loss: 1.400 |  Val. Acc: 50.22% 

	Train Loss: 1.337 | Train Acc: 58.04%
	 Val. Loss: 1.400 |  Val. Acc: 49.83% 

	Train Loss: 1.326 | Train Acc: 58.97%
	 Val. Loss: 1.402 |  Val. Acc: 49.61% 

	Train Loss: 1.318 | Train Acc: 59.72%
	 Val. Loss: 1.405 |  Val. Acc: 49.05% 
   '''
   ### 10 example inputs from the test/validation data
   Tweet                       prediction           Actual
Action mechanical    Neutral         Neutral
"
The story moldy obvious    very Negative         very Negative
"
The movie fails portray literarily talented notorious subject anything much dirty old man    very Negative         very Negative
"
A comprehensive provocative film -- one pushes boundary biography    very Negative         very Negative
"
fear dot com rambling disconnected never builds suspense    very Negative         very Negative
"
Frustratingly    Negative         Negative
"
Director Brian Levant    very Negative         very Negative
"
Purposefully shocking eroticized gore    Neutral         Neutral
"
Though never rises full potential film    Neutral         Neutral
"
soul 's lacking every character movie    very Negative         very Negative
"
