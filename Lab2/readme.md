# Lab 2 
## Introduction
In this laboratory, you can find several experiments regarding Large Language Models 
(e.g. GPT and BERT).
## Exercise 1: Warming Up
In this first exercise we implemented a GPT from scratch based on  [Andrej Karpathy video](https://www.youtube.com/watch?v=kCc8FmEb1nY)
and trained on Inferno by Dante to generate text in his style. The file containing text for training is avaliable at this
[link](https://archive.org/stream/ladivinacommedia00997gut/1ddcd09.txt)
We trained the model for 5000 epochs and generate some text at the end of training.
The code for this exercise is in `Exercise1`.
The results obtained are in `./results/dante_exercise1/output.txt`. Here we show a passage:
>Dopo lunga, lo maestro e argento ad Achigna<br>
  in quella focata viglia fia mano.<br><br>
Poi si volse in se' con ascolta,<br>
  e ancor lora piu` dov'intramente acque.<br><br>
Questa li era con la scesarita maia<br>
  di qua da l'altra bollon fin ti fanno,<br>
  che non pur potea l'era la tosca.

## Exercise 2: Working with Real LLMs

In this exercise we used the [Hugging Face](https://huggingface.co/) model and dataset 
ecosystem to access a *huge* variety of pre-trained transformer models.
The code is in `Exercise2`
First thing, we used Hugging Face GPT-2 model using its tokenizer to encode text 
into sub-word tokens. 
We compared the length of input with the encoded sequence length.
Results are shown in the table below:

|              <!-- -->               | <!-- -->        |
|:-----------------------------------:|:---------------:|
| # of characters in Dante's Inferno  | 186983       |
| Tokenized length of Dante's Inferno | 79225        | 
|              **Ratio**              |  0.4237    | 


Then we instantiated a pre-trained `GPT2LMHeadModel` and use the [`generate()`](https://huggingface.co/docs/transformers/v4.27.2/en/main_classes/text_generation#transformers.GenerationMixin.generate) 
method to generate text from a prompt.
The prompt for the generation was 
    
    Alice and Bob

We experimented several parameters:
* `do_sample`:  this parameter enables several decoding strategies that select the 
next token from the probability distribution over the entire vocabulary.
* `temperature`:  to control the randomness



`do_sample=False`
>Alice and Bob are both in the same room.<br>
"I'm not sure if you're going to be able to get out of here," Bob says. <br>
"I'm not sure if you're going to be able to get out of here." <br>
"I'm not sure if you're going to be able to get out of here," Bob says. <br>
"I'm not sure if you're going to be able to get out of here."

`do_sample=True, temperature=0.1`
>Alice and Bob are both in the same room.<br>
The two girls are in a room with a large table and a large table with a large table.<br>
The two girls are sitting on the table.<br>
The two girls are sitting on the table.<br>
The two girls are sitting on the table.<br>
The two girls are sitting on the table.<br>
The two girls are sitting on the table.<br>
The two girls are sitting on the table.<br>

`do_sample=True, temperature=0.5`

>Alice and Bob's parents, who lived outside of town, are now living in the same house. <br>
The couple's two daughters, ages 7 and 8, are also among the first to be diagnosed with autism. <br>
The children are all from the same family, but they are being treated for the same condition. <br>
A spokesman for the Department of Social Services said the couple's condition is not life-threatening. <br>
It is not known if the children will be able to return to school.

`do_sample=True, temperature=1`
>Alice and Bob, and my own character is that of a woman, with a husband and 
a daughter. It seems to me in some ways he's going to show his very core strength
and I think that's how our characters will start out.
But now we are in a new place, we are no longer an old boy character. At least, 
that's what I think is going to happen. Because all the characters have changed, they have different styles, different roles, different personalities, and different tastes

To get a satisfying output we need the `do_sample=True` parameter set to `True` and a fairly high temperature coefficient. 

## Exercise 3.1: Training a Text Classifier 
In this exercise we exploit a pretrained BERT as the backbone for the text classifier.
### Tweet_Eval Dataset
I have chosen to use `tweet_eval` dataset from text classification datasets on [Hugging Face](https://huggingface.co/datasets?task_categories=task_categories:text-classification&sort=downloads)
which is made up of 5052 tweets from twitter divided in 4 classes:
* anger (43%)
* sadness (26.3%)
* joy (21.7%)
* optimism (9%)

### The experiment
The model trained to perform text classification uses DistilBERT as backbone and 
three fully connected layers (with a ReLU as activations) as classification head.

First we create a baseline for this problem using the LLM 
*exclusively* as a feature extractor and then trained the classification head.

Then we fine-tuned the entire model for 1 epoch.
The code is in `Exercise3_1.py`.
The table below show the results obtained:

|        Training Strategy         | Accuracy |
|:--------------------------------:|:--------:|
|           No training            |  0.2519  |
| Training the classification head |  0.6890  | 
|   Fine-tuning the entire model   |  0.7861  | 

In the first scenario, where the classification head is initialized at random 
the model outputs random predictions. 
The second and third cases show good performances, but  fine-tuning
the entire models leads to the best performance.

## Exercise 3.2: Training a Question Answering Model 
Here we used the SWAG dataset to train to answer contextualized multiple-choice 
questions.
The code is in `Exercise3_2.py`. 

Unlike the previous exercise, here we exploit the `Trainer` class which provides an API 
for feature-complete training in PyTorch.

After fine-tuning, the model is able to answer simple questions such as:
>What musical genre was born in the United States?<br>
0: Jazz<br>
1: Baroque<br>
2: Classical<br>
Model answer: Jazz

The other interactions with our model can be found in `./results/exercise3_2/interactions.txt`

