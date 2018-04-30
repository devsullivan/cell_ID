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


def write_output(img_list,label_list):

def main():
    
    #Parameters 
    featpath = '/Users/devinsullivanMBP/cell_ID_hackathon/features/'
    labelpath = '/Users/devinsullivanMBP/cell_ID_hackathon/training_upload.csv'
    validation_featpath = '/Users/devinsullivanMBP/cell_ID_hackathon/validation_features2/' 
    
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
    feats = np.empty((1,2233))

    
    #go through files 
    for filename in glob.iglob(featpath+'**/*features.csv', recursive=True):
       imgname = os.path.basename(os.path.dirname(filename))
       #get features and number of lines in the file
       currfeats = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=0)
       print(np.shape(currfeats))
       feats = np.vstack((feats,currfeats))
       currlines = np.shape(currfeats)
       num_lines.append(currlines[0])
       name_list.append([imgname]*num_lines[-1])
       label_list.append([cell_lines[imgname]]*num_lines[-1])

    print(np.shape(feats))
    #Try a classifier for literally only the number of lines (cells)

    num_lines = np.asarray(num_lines)
    num_lines = num_lines.reshape(-1,1)
    num_imgs = len(num_lines) 
    #init
    numline_classifier = svm.SVC()
    #train
    numline_classifier.fit(feats,label_list)
    #predict
    train_accuracy = np.zeros(num_imgs)
    for i,(cellcount,label) in enumerate(zip(feats,label_list)):
        print(label)
        print(cellcount)
        curr_pred = numline_classifier.predict(cellcount[0])
        train_accuracy[i] = curr_pred==label
    print('training accuracy: ')
    print(np.sum(train_accuracy)/num_imgs)
    
    #read the validation set
    num_lines_validation = []
    testfeats = np.empty((1,2233))
    pred_list = []
    name_list_validation = []
    for filename in glob.iglob(validation_featpath+'**/*features.csv', recursive=True):
       imgname = os.path.basename(os.path.dirname(filename))
       #get features and number of lines in the file
       curr_testfeats = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=0)
       print(np.shape(curr_testfeats))
       testfeats = np.vstack((testfeats,curr_testfeats))
       currlines = np.shape(curr_testfeats)
       num_lines_validation.append(currlines[0])
       name_list_validation.append([imgname]*num_lines[-1])
       #predict validation labels 
       curr_pred = numline_classifier.predict(testfeats)
       pred_list.append(curr_pred)
       name_list.append(imgname)
       #label_list.append(cell_lines[imgname])
    num_lines_validation = np.asarray(num_lines_validation)
    
    
    
    #write output 
    
    cell_lines = {}
    with open(labelpath,'r') as validationfile:
        labelreader = csv.reader(labelfile,delimiter=',')
        labelreader.__next__()
        for row in labelreader:
            cell_lines[row[0]]= row[1]

    
    

if __name__ == "__main__":
    main()