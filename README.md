# Sentiment Analysis Project

[URL to the Kaggle Repo](https://www.kaggle.com/kazanova/sentiment140) - Download the training file and rename it to 'train.csv' to run the notebook. If you want to skip ahead to when the preprocessing is completed and reproduce the same results, extract the 'Final.csv' file from the compressed zip version included with the repository. With both these csv files in the folder, you can run the entire notebook! 

### Introduction 
For this project we used the sentiment140 dataset from Kaggle. This dataset was compiled using the twitter API and has 1.6 million tweets. The data has also been annotated with a sentiment polarity score between 0 to 4, a score close to 0 indicating a negative sentiment and a score closer to 4 indicating a positive sentiment.

These are the fields being used in the dataset:-
target:	 The polarity of the tweet. (0 = negative, 2 = neutral, 4 = positive)
ids:		 The id of the tweet.
date: 	 The date of the tweet. 
flag:	 The query. If there is no query, then this value is NO_QUERY.
user:	 The user that tweeted. 
text: 	 The text of the tweet.

### Preprocessing 
We first read in the data using pandas and had to restructure the data due to the columns not having any names. We then took a chunk of 300000 rows from the data due to computational constraints to work on the model. First thing we did was change the 0-4 sentiment structure with a text labelling system and made 0's Negative and 4's as Positive.

Next we cleaned the tweets by removing hashtags and all punctuations, except Apostrophes. To do this we used Regular Expressions to seek out hashtags and @s in the tweets and remove the associated data with it.

### Data visualization 
Once the preprocessing was complete we generated word clouds for positive and negative tweets. To compile this we used Pointwise Mutual Information (PMI) scores and used those scores to generate word clouds. 

### Model Building
We split the data into train test splits using an 80-20 train to test split ratio. We then used Spacy to tokenize the tweets and fed the tokens into a few linear models - SGD, Random Forest, Passive Aggressive model and the Ngram models to begin with. The Ngram model performed the best and that was then subjected to hyperparameter tuning using DASK. The Tuned Ngram model performed the best with the scores shown in the result section.

Then we tried adding negations and word combinations to the tweets to see if they would improve the performance. Since these were also kind of similar to the unibigrams technique, we couldn't use them with the NGram model but had to train on the baseline SGD. They performed well with the word combination model performing equally as good as the Ngram Model with a score of 0.81. 

We then tried to improve model performance using Margin based dataset filtering, but there was no clear indication of a margin because it was a 2 class classification therefore there was only one score for the decision function. And both positive and negative sentiments had similar scores so a Margin could not be arrived at for the calculation.

Next we compared the best performing SGD model to a Distilbert model. The distilbert was fit on the training and test data and several values were tried for the epochs and the batch sizes. The best performing distilbert model had an F1 of 0.84 making it a fair bit better than the SGD models. 

### Model explanation 
To see which parts of the classification we got correct and wrong, we used LIME to explain how the model made its classifications. 

With the lime text explainers we could see clearly the patterns that we observed in the tweets that we classified correctly. Words like Sad and Hope contributed very highly to the sentiment score being skewed towards that positive or negative direction. 

With the classifications we got wrong, we could see an example where the Tag assigned seemed wrong compared to the tag that we actually got from the model. The explanation seemed to indicate that the model was indeed correct and this could be an indication of poor labelling of the training data. 

The other example where the tweet itself seemed ambiguous also indicated that the model got confused because there was no real positive or negative intent behind the tweet at all. All in all the model did a good job

Finally a confusion matrix was generated to see the performance of the classifier.

### Result
These were the models that were trained and their F1 scores respectively. 

| Model | F1 |
| ------ | ------- |
| Distilbert | 0.838120 | 
| Word Combination SGD | 0.809057 |
| Optimised Ngram | 0.806281	|
| Baseline Ngram| 0.801524 |
| Negation SGD | 0.797902 | 
| SGD Baseline | 0.779827 | 
| Random Forest | 0.778118 | 
| Passive Aggressive model | 0.738672 | 


