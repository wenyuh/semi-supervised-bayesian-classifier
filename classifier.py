# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:04:38 2018

@author: Alexandre Boyker
"""
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from helper import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class NaiveBayesSemiSupervised(object):
    
    def __init__(self, max_features=None, max_rounds=30, tolerance=1e-6):
        
        self.max_features = max_features
        self.n_labels = 0
        self.max_rounds = max_rounds
        self.tolerance = tolerance
        
    def fit(self, X_supervised, X_unsupervised, y_supervised):
        
        count_vec = CountVectorizer(max_features=self.max_features)
        count_vec.fit(X_supervised + X_unsupervised)
        self.n_labels = len(set(y_supervised))
        
        if self.max_features is None:
            
            self.max_features = len(count_vec.vocabulary_ )
            
        X_supervised = np.asarray(count_vec.transform(X_supervised).todense())
        X_unsupervised = np.asarray(count_vec.transform(X_unsupervised).todense())
        
        self.train(X_supervised, y_supervised)

        predi = self.predict(X_supervised)

        old_likelihood = 1
        
        while self.max_rounds > 0:
            
            self.max_rounds -= 1
            predi = self.predict(X_unsupervised)
            self.train(X_unsupervised, predi)
            predi = self.predict(X_supervised)
            total_likelihood = self.get_log_likelihood( X_supervised, X_unsupervised, y_supervised)
            print("total likelihood: {}".format(total_likelihood))
            if self._stopping_time(old_likelihood, total_likelihood):
                break
            old_likelihood = total_likelihood.copy()
            
    def _stopping_time(self, old_likelihood, new_likelihood):
        
        relative_change = np.absolute((new_likelihood-old_likelihood)/old_likelihood) 
        
        if (relative_change < self.tolerance):
        
            print("stopping time")
            return True
        
        else:
            
            return False
    
    def get_log_likelihood(self, X_supervised, X_unsupervised, y_supervised):
        
        unsupervised_term = np.sum(self._predict_proba_unormalized(X_unsupervised), axis=1)
        unsupervised_term = np.sum(np.log(unsupervised_term))
        supervised_term = self._predict_proba_unormalized(X_supervised)
        supervised_term = np.take(supervised_term, y_supervised)
        supervised_term = np.sum(np.log(supervised_term))
        total_likelihood = supervised_term + unsupervised_term
        
        return total_likelihood
        
    def word_proba(self, X, y, c):
        
        numerator = 1 + np.sum( X[np.equal( y, c )], axis=0)
        denominator = self.max_features + np.sum( X[ np.equal( y, c)])
        
        return np.squeeze(numerator)/denominator
    
    
    def class_proba(self, X, y, c):
        
        numerator = 1 + np.sum( np.equal( y, c) , axis=0)
        denominator = X.shape[0] + self.n_labels
        return numerator/denominator
        
    def train(self, X, y):
        
        word_proba_array = np.zeros(( self.max_features, self.n_labels))
        
        for c in range(self.n_labels):
            
            word_proba_array[:,c] = self.word_proba( X, y, c)
            
        labels_proba_array = np.zeros(self.n_labels)
        
        for c in range(self.n_labels):
            
            labels_proba_array[c] = self.class_proba( X, y, c)
            
        self.word_proba_array = word_proba_array
        self.labels_proba_array = labels_proba_array
    
    
    def _predict_proba_unormalized(self, X_test):
        
        proba_array_unormalized = np.zeros((X_test.shape[0], self.n_labels))
        
        for c in range(self.n_labels):
            
            temp = np.power(np.tile(self.word_proba_array[:,c], (X_test.shape[0] ,1)), X_test)
            proba_array_unormalized[:,c] = self.labels_proba_array[c] * np.prod(temp, axis=1)
            
        return proba_array_unormalized
    
    def predict_proba(self, X_test):
        
        proba_array_unormalized = self._predict_proba_unormalized(X_test)
        proba_array = np.true_divide(proba_array_unormalized,np.sum(proba_array_unormalized, axis=1)[:, np.newaxis])
        return proba_array
    
    def predict(self, X_test):
        
        return np.argmax(self.predict_proba( X_test), axis=1)
        
        
    

    
    
    
    