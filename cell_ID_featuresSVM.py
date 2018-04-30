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
from scipy import stats
from sklearn.decomposition import PCA

    

def main():
    
    #Parameters 
    featpath = '/Users/devinsullivanMBP/cell_ID_hackathon/training_features/'
    labelpath = '/Users/devinsullivanMBP/cell_ID_hackathon/training_upload.csv'
    validation_featpath = '/Users/devinsullivanMBP/cell_ID_hackathon/validation_features/' 
    outpath = '/Users/devinsullivanMBP/cell_ID_hackathon/validation_predictions.csv'
    
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
    #feats = np.empty((1,2233))

    
    #go through files 
    print('reading feats')
    for filename in glob.iglob(featpath+'**/*features.csv', recursive=True):
       imgname = os.path.basename(os.path.dirname(filename))
       #get features and number of lines in the file
       currfeats = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=0)
       if 'feats' in locals():
           feats = np.vstack((feats,currfeats))
       else:
           feats = currfeats
               
       currlines = np.shape(currfeats)
       num_lines.append(currlines[0])
       name_list.append([imgname]*num_lines[-1])
       currnames = [cell_lines[imgname]]*num_lines[-1]
       label_list.extend(currnames)
       
    print('normailization and pca')
    #normalize features with zscore 
    mufeats = np.mean(feats,axis=0)
    stdfeats = np.std(feats,axis=0)
    zfeats = (feats-mufeats)/stdfeats#stats.zscore(feats)
    #remove any columns that are always nan
    nan_cols = ~np.any(np.isnan(zfeats), axis=0)
    zfeats = zfeats[:,nan_cols]
    inf_cols = ~np.any(np.isinf(zfeats), axis=0)
    zfeats = zfeats[:,inf_cols]
    
    #perform Principal Components Analysis to reduce dimension
    #randomly choosing 5 components for now. 
    pca = PCA(n_components=5)
    pca.fit(zfeats)
    zf_pca5 = pca.transform(zfeats)
    
    print(np.shape(zf_pca5))
    
    #Try a classifier for literally only the number of lines (cells)
    num_lines = np.asarray(num_lines)
    num_lines = num_lines.reshape(-1,1)
    num_imgs = len(num_lines) 

    #init svm
    features_classifier = svm.SVC()
    #numline_classifier = svm.SVC()
    #train
    print(len(label_list))
    features_classifier.fit(zf_pca5,label_list)
    #numline_classifier.fit(num_lines,label_list)
    #predict
    train_accuracy = np.zeros(num_imgs)
    pred = features_classifier.predict(zf_pca5)
    acc = pred==label_list
    print('training accuracy per-cell: ')
    print(np.sum(acc)/np.sum(num_lines))
    print(pred)

    print(breakme)
    #read the validation set
    num_lines_validation = []
    #testfeats = np.empty((1,2233))
    pred_list = []
    name_list_validation = []
    with open(outpath, 'w') as outfile:
        resultwriter = csv.writer(outfile,delimiter=',')
        for filename in glob.iglob(validation_featpath+'*features.csv', recursive=True):
           imgname = os.path.basename(filename)
           imgname = imgname.split('_')[0]
           print(imgname)
           #get features and number of lines in the file
           curr_testfeats = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=0)
           #normalize with training z-score
           ztestfeats = (curr_testfeats-mufeats)/stdfeats
           #remove nan *then* inf (order is important)
           ztestfeats = ztestfeats[:,nan_cols]
           ztestfeats = ztestfeats[:,inf_cols]
           ztest_pca5 = pca.transform(ztestfeats)
           #predict validation labels 
           cell_pred = features_classifier.predict(ztest_pca5)
           print(cell_pred)
           #grab the mode of the predictions as the overall prediction
           curr_pred = stats.mode(cell_pred)
           print(curr_pred[0][0])
           
           #write to file 
           resultwriter.writerow([imgname,curr_pred[0][0]])

    
    
    
    #write output 
    #write_output(name_list_validation,pred_list,outpath)
    
    
    

if __name__ == "__main__":
    main()