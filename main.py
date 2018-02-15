# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 11:32:47 2018

@author: Alexandre Boyker
"""


from helper import StsaParser, plot_confusion_matrix
from classifier import NaiveBayesSemiSupervised
import os
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    
    stsa_parser = StsaParser()

    data, labels = stsa_parser.get_data(os.path.join('stsa','stsa-train.txt'))
    
    X_supervised, X_unsupervised, y_supervised, y_unsupervised = train_test_split(data, labels, test_size=0.8)
    
    clf = NaiveBayesSemiSupervised(max_features=7000)
    
    clf.train(X_supervised, X_unsupervised, y_supervised)
    

if __name__ == '__main__':
    
    main()