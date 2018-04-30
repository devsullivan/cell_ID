#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 13:20:10 2018

@author: devinsullivanMBP
"""

import numpy as np 
import matplotlib as plt 
import csv
import os
import glob 
from sklearn import svm 
from sklearn.ensemble import RandomForestClassifier

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def main():
    
    #Parameters 
    featpath = '/Users/devinsullivanMBP/cell_ID_hackathon/features/'
    labelpath = '/Users/devinsullivanMBP/cell_ID_hackathon/training_upload.csv'
    validation_featpath = '/Users/devinsullivanMBP/cell_ID_hackathon/validation_features/' 
    
    cell_lines = {}
    with open(labelpath,'r') as labelfile:
        labelreader = csv.reader(labelfile,delimiter=',')
        labelreader.__next__()
        for row in labelreader:
            cell_lines[row[0]]= row[1]
    #print(labels[1][0])
    #print(cell_lines)

    
    #initialize
    num_lines = []
    name_list = []
    label_list = []
    
    #go through files 
    for filename in glob.iglob(featpath+'**/*features.csv', recursive=True):
       imgname = os.path.basename(os.path.dirname(filename))
       #get number of lines in the file
       num_lines.append(file_len(filename))
       name_list.append(imgname)
       label_list.append(cell_lines[imgname])


    #Try a classifier for literally only the number of lines (cells)

    num_lines = np.asarray(num_lines)
    num_lines = num_lines.reshape(-1,1)
    num_imgs = len(num_lines) 
    #init
    numline_classifier = svm.SVC()
    #train
    numline_classifier.fit(num_lines,label_list)
    #predict
    train_accuracy = np.zeros(num_imgs)
    for i,(cellcount,label) in enumerate(zip(num_lines,label_list)):
        print(label)
        print(cellcount)
        curr_pred = numline_classifier.predict(cellcount[0])
        train_accuracy[i] = curr_pred==label
    print('training accuracy: ')
    print(np.sum(train_accuracy)/num_imgs)
    
    #read the validation set
    num_lines_validation = []
    pred_list = []
    for filename in glob.iglob(validation_featpath+'**/*features.csv', recursive=True):
       imgname = os.path.basename(os.path.dirname(filename))
       #get number of lines in the file
       num_lines_validation.append(file_len(filename))
       #predict validation labels 
       curr_pred = numline_classifier.predict(num_lines_validation[-1])
       pred_list.append(curr_pred)
       name_list.append(imgname)
       #label_list.append(cell_lines[imgname])
    num_lines_validation = np.asarray(num_lines_validation)
    
    
    
    
    #get the feature file names 
    cell_lines = {}
    with open(labelpath,'r') as validationfile:
        labelreader = csv.reader(labelfile,delimiter=',')
        labelreader.__next__()
        for row in labelreader:
            cell_lines[row[0]]= row[1]

    
    

if __name__ == "__main__":
    main()