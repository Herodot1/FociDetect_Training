# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:25:28 2019

@author: Tim Hohmann et al. - "Evaluation of machine learning models for 
automatic detection of DNA double strand breaks after irradiation using a gH2AX 
foci assay", PLOS One, 2020
"""

# diverse helper functions for ease of use in model training and testing

# import numpy:
import numpy as np
# import pandas:
import pandas as pd

# import packages related to (pre-) processing of data and model results:
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

# import stuff for resampling data:
from random import sample

def get_best_params(x_vals,y_vals,model,parameters,var_max = 0.99):
    # rescale data
    s = StandardScaler()
    x_train_transformed = s.fit_transform(x_vals)
    # do PCA to reduce parameter number:
    p = PCA()
    p.fit(x_train_transformed)
    x_train_transformed = p.transform(x_train_transformed)           
    # cumulated sum of variance explained. take only data explaining 99% of
    # variance:
    var_exp = np.cumsum(p.explained_variance_ratio_)
    idx = var_exp<=var_max
    x_train_transformed = x_train_transformed[:,idx]
    # make a grid search for parameter optimization:
    clf = GridSearchCV(model, parameters, cv = 3)
    clf.fit(x_train_transformed, y_vals)
    
    return clf.best_params_

def image_pipeline(x_data,y_data,removed_im = [],var_max = 0.99, sampling = 1): 

    '''
    Input: 
    x_data, y_data  = list of lists containing feature data (x_data) or 
    the respective classifications (y_data) for each image. 
    x/y_data[image_number] must return x/y_data of the respective image.
    removed_im      = list of image numbers to be left out
    var_max         = use only principal components explaining var_max of the 
    total variance. 0<=var_max<=1
    sample          = proportion of each image used for fitting etc (also 
                    equals the relative amount of data in the output)
    Output:
    x/y_out         = merged list containing all images but the ones left out. denote 
    this merges all images.  
    idx             = boolean used for taking only important principal 
    components
    '''
    x_vals, y_vals = leave_out_ims(x_data,y_data,removed_im, sampling)
    
    x_train = pd.DataFrame(x_vals)
    y_train = pd.DataFrame(y_vals)      
    # rescale data
    s1 = StandardScaler()
    x_train = s1.fit_transform(x_train)
    # do PCA to reduce parameter number:
    p = PCA()
    p.fit(x_train)
    x_train = p.transform(x_train)
    # cumulated sum of variance explained. take only data explaining 95% of
    # variance
    var_exp = np.cumsum(p.explained_variance_ratio_)
    idx = var_exp<=var_max
    x_train = x_train[:,idx]
    s2 = StandardScaler()
    x_train = s2.fit_transform(x_train)
    mnmx = MinMaxScaler(feature_range=(0, 1))
    x_train = mnmx.fit_transform(x_train)
    return x_train, y_train, idx, s1, s2, p, mnmx

def leave_out_ims(x_data,y_data,one_out, sampling = 1): 
    '''
    Input: 
    x_data, y_data  = list of lists containing feature data (x_data) or 
    the respective classifications (y_data) for each image. 
    x/y_data[image_number] must return x/y_data of the respective image.
    one_out         = list of image numbers to be left out
    sample          = proportion of each image used for fitting etc (also 
                    equals the relative amount of data in the output)
    Output:
    x/y_out         = merged list containing all images but the ones left out. denote 
    this merges all images.
    '''
    x_out = []
    y_out = []
    
    # catch error if sampling is set larger than 1
    if sampling > 1:
        sampling = 1        
    
    for im in range(len(x_data)):
        if sampling != 1:
            idx = sample(range(len(x_data[im])), round(len(x_data[im])*sampling))
        else:
            idx = list(range(len(x_data[im])))       
            
        if im not in one_out:               
            x_temp = [x_data[im][i].copy() for i in idx]                    
            x_out.extend(x_temp.copy())
                
            y_temp = [y_data[im][i] for i in idx]                    
            y_out.extend(y_temp.copy())         
            
    return x_out, y_out 

    
    
    
    
    
    
    
    