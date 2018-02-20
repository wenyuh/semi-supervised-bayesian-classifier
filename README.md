# semi-supervised-bayesian-classifier
Modified Naive Bayes for semi-supervised learning 

TODO: Finish Readme
## Overview

The Naive Bayes classifier is known to work well on NLP tasks such as documents classification, after the documents have been vectorized using for instance a bag-of-words or word count embedding. However, in the presence of a small number of labelled documents and a large number of unlabelled documents, would it be possible to 'tweak' regular Naive Bayes in order to take advantage of the vast amount of unlabelled data? We consider here the results of Nigam et al. (Semi-supervised text classification using EM, available in the book simply called 'Semi-supervised learning').

## A generative model for documents

We build a generative model for documents, whose intuition is quite simple: 

> 1) if we have m classes (m labels to predict), we throw a m-sided biased dice to predict the class of that document

3) we throw the V-sided biased dice related to this label (V is the size of the vocabulary) |x_i| times, where |x_i| is the length of document i

3) count the number of times that each word appears to get a 'word count' representation
