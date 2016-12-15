#!/usr/bin/python
"""
Demo for the troll_classifier.py

Assumes that a model exists
"""

import troll_classifier

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import sys
import re #regular expressions for punctuation stripping -TS
from sklearn.externals import joblib #save models
from sklearn import svm

import settings


def main():
    words_file = 'oldProject/reddit_comments/feature_words.txt'
    feature_words = troll_classifier.parse_file(words_file)
    model = joblib.load('TrollModel.pk1')
    comment = raw_input('Enter a comment to check: ')

    result = troll_classifier.it_is_a_troll(model, feature_words, comment)
    if result:
        print 'This is a troll comment!'
    else:
        print 'This is not a troll comment.'

if __name__ == '__main__':
    main()

