# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 19:08:39 2019
@author: Tim Hohmann et al. - "Evaluation of machine learning models for 
automatic detection of DNA double strand breaks after irradiation using a gH2AX 
foci assay", PLOS One, 2020
"""

# main file for training machine learning models using previously labeled data

###############################################################################
# Parameters and file path that have to be set manually:

# Parameters:
# min area of nucleus
min_area = 4000
# color channel of nucleus. 0 = red, 1 = grenn, 2 = blue. for grayscale images
# this value is ignored.
nuc_chan = 2
# color channel of foci. 0 = red, 1 = grenn, 2 = blue. for grayscale images
# this value is ignored.
foci_chan = 1
# color channel of marked image. 0 = red, 1 = grenn, 2 = blue. corresponds to 
# the color of the markings in the manually labeled foci images
mark_chan = 0

# adjust image size - might be usefull to save calculation time. needs to be 
# identical for foci and nucleus images
# image rescale factor:
rescale_factor = 1.0
# take only those PCA components cumulatively explaining var_max of the variance
# 0<var_max<=1. 
var_max = 0.95
# randomly sample a proportion of the training data from each image (0<sampling<=1). 
# speeds up training process if smaller than 1
sampling = 1

# used filter sizes
filt_range = [2,3,4,5,8,10,15,20,25,30,35]
# scaling range for frangi filter
sc_range = list(range(2,11))
#frequency range for gabor filter
freq = [0.08,0.10,0.13,0.16,0.2]

# Name used for saving the trained model and related images:
model_name = "MLP"
# directory containing the foci images:
im_path_foci = "D:\\Sample Images\\foci"
# directory containing the manually labeled foci images:
im_path_foci_marked = "D:\\Sample Images\\foci_marked"
# directory containing the nucleus images:
im_path_dapi = "D:\\Sample Images\\dapi"

###############################################################################
###############################################################################
###############################################################################
# turn of warnings, this is especially annoying with sklearn stuff
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Get packages:
# For directory, file handling etc.
import os
import sys
# import pandas:
import pandas as pd
# import numpy
import numpy as np
# For image analysis:
from skimage.io import imread, imshow
from skimage.io import imsave
from skimage.transform import rescale
# import packages related to (pre-) processing of data and model results:
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
# import model packages:
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,  BaggingClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score

sys.path.append(os.getcwd())
main_file_path = os.getcwd()
# self written functions:
from FociDetectImgAnalysis import get_nuclei, GoToFiles
from GetFociData import get_labeled_data
from GetBestParams import image_pipeline
###############################################################################
# Start analysis of dapi images
# go to nucleus folder:
os.chdir(im_path_dapi)
print("Analyzing nucleus images ...")
# go to the directory containing the foci images:
#im_path = os.getcwd()+"\Sample Images\foci"
#os.chdir(im_path)

# start reading in image files:
stats = []        

# get file names
save_path = "Single Nuclei"        
files = GoToFiles(im_path_dapi,save_path)
        
for file_num in range(len(files)):
    file_name, extension = os.path.splitext(files[file_num])
    # print(file_name + "   " + extension)
    if extension in [".png",".tif",".jpg",".bmp"]:         
        # read image:                 
        image = imread(files[file_num])
        image = rescale(image, rescale_factor, order=1,preserve_range = True)          
        image = np.uint8(image) 
        #imshow(image)       
        # get region props of the blue channel:
        if(len(image.shape)<3):
            stats.append(get_nuclei(image[:,:],file_name))
        else:
            stats.append(get_nuclei(image[:,:,nuc_chan],file_name))
   
# Get x and y data for model training and the coordinate for each image:
# y_data is boolean with True were pixel was marked as foci and false otherwise
x_data, y_data, coords = get_labeled_data(im_path_foci,stats,im_path_foci_marked,filt_range,freq,sc_range,rescale_factor,foci_chan, mark_chan)  
# When done with everything go back to the main folder:
os.chdir(main_file_path)



###############################################################################
# This part is for model training assuming "get_labeled_data" was ran successfully

# get trainingsdata:
x_train_transformed, y_train, idx, s1, s2, p, mnmx = image_pipeline(x_data,y_data,removed_im = [],var_max = var_max, sampling = sampling)

# Chose the model to train:
# neural network:
model =  MLPClassifier(alpha=0.1, batch_size = 2000, learning_rate = "adaptive", learning_rate_init = 0.1, max_iter = 300, tol = 10**-4, early_stopping = True)
parameters = {'batch_size':[100,500,1000,2000,3000,4000,5000],'alpha':[10**-4,10**-3,10**-2,10**-1,1]}

# random forest:
# clf = RandomForestClassifier(criterion = "entropy",min_weight_fraction_leaf = 0.005, n_estimators = 15,max_depth = 50, min_samples_leaf = 10,min_samples_split = 100, n_jobs = -1)
# model =  AdaBoostClassifier(base_estimator = clf, n_estimators=5)
# parameters = {'base_estimator__min_weight_fraction_leaf':[0.0001,0.001,0.005],'base_estimator__n_estimators':[5,10,15,20],'base_estimator__min_samples_leaf':[10,20,100]}

# complement naive bayes:
# clf = ComplementNB(alpha = 0.0, norm = True)
# model =  AdaBoostClassifier(base_estimator = clf, n_estimators=15)
# parameters = {'base_estimator__alpha': [0,0.01,0.02,0.03,0.04,0.05,0.06], 'base_estimator__norm': [True, False]}

# support vector machine:
# linear svm
# clf = LinearSVC(penalty = "l2", loss = "hinge", C = 2, class_weight = "balanced", max_iter = 5000)
# model =  AdaBoostClassifier(base_estimator = clf, n_estimators=5,algorithm='SAMME')
# parameters = {"base_estimator__C": [0.1,0.3,0.6,1,2,3]}

print("Performing grid search ...")
# get best model parameters:
clf = GridSearchCV(model, parameters, cv = 3)
clf.fit(x_train_transformed, y_train)

###############################################################################
# train models based on on all but one of the  images and test on the remaining 
# one. Do this for all combinations of images.
# Save images and some resulting statistics.

# save path:
save_path = im_path_foci + "\Results Model Validation"
# set model:
# neural network:
# model =  MLPClassifier(alpha=0.1, batch_size = 2000, learning_rate = "adaptive", learning_rate_init = 0.1, max_iter = 300, tol = 10**-4, early_stopping = True)
model = clf.best_estimator_
im_stats = []    
# create data sets leaving out one image:
print("Training model (leave one out) ...")
for im in range(len(x_data)):
    print("Current Image:" + str(im+1))
    removed_im = [im] 
    x_train_transformed, y_train, idx, s1, s2, p, mnmx = image_pipeline(x_data,y_data,removed_im,var_max = 0.95, sampling = 1)
    
    # use some defined model and train it with the x-featues:
    model.fit(x_train_transformed, y_train)         
    # create variables for test image:
    x_vals_im = pd.DataFrame(x_data[removed_im[0]])
    y_vals_im = pd.DataFrame(y_data[removed_im[0]])    
    # rescale data
    x_image_transformed = s1.transform(x_vals_im)
    # do PCA to reduce parameter number:
    x_image_transformed = p.transform(x_image_transformed)
    # cumulated sum of variance explained. take only data explaining 95% of
    # variance
    x_image_transformed = x_image_transformed[:,idx]
    x_image_transformed = s2.transform(x_image_transformed)
    x_image_transformed = mnmx.transform(x_image_transformed)
    #predict labels:
    y_test_pred = model.predict(x_image_transformed)     
    # prediction and confusion matrix  
    conf_mat = confusion_matrix(y_vals_im, y_test_pred)
    v1 = model.score(x_image_transformed, y_vals_im)
    v2 = precision_score(y_vals_im, y_test_pred) # sensitivity
    v3 = recall_score(y_vals_im, y_test_pred)
    
    im_stats.extend(["Image_"+str(removed_im[0]),conf_mat,v1,v2,v3])
    
    # apply model to test data:
    files = GoToFiles(im_path_foci,"\Results Model Validation")
    image = imread(files[removed_im[0]])  
    temp_var = image.copy()
    
    for i in range(len(y_test_pred)):    
        if y_test_pred[i] == True:
            temp_var[coords[removed_im[0]][i][0],coords[removed_im[0]][i][1]] = 255              
    files = GoToFiles(save_path)
    save_name =   "Im_"+ str(removed_im[0]+1) + "_" + model_name + ".png"     
    imsave(save_name,temp_var)      

    
# write the statistics data for each analyzed image:
import csv
files = GoToFiles(save_path)
with open("image_statistics.txt", 'w') as output:
     wr = csv.writer(output,lineterminator='\n')     
     for val in im_stats:
         wr.writerow([val])


###############################################################################
# train specific model on whole dataset and save fitted model.
# save path:
print("Training model (all images) ...")
save_path = im_path_foci + "\Trained Models" + "\\" + model_name
# set model:
# neural network:
# model =  MLPClassifier(alpha=0.1, batch_size = 2000, learning_rate = "adaptive", learning_rate_init = 0.1, max_iter = 300, tol = 10**-4, early_stopping = True)
model = clf.best_estimator_
removed_im = [] 
x_train_transformed, y_train, idx, s1, s2, p, mnmx = image_pipeline(x_data,y_data,removed_im,var_max = 0.95, sampling = 1)

files = GoToFiles(im_path_foci,"\Trained Models" + "\\" + model_name)
GoToFiles(save_path)
# save scalings:
save_name = "PCA.p"   
import pickle
with open(save_name, "wb") as fp:   #Pickling  
    pickle.dump(p, fp)

save_name ="STD_Scaler1.p"   
with open(save_name, "wb") as fp:   #Pickling  
    pickle.dump(s1, fp)
    
save_name = "STD_Scaler2.p"   
with open(save_name, "wb") as fp:   #Pickling  
    pickle.dump(s2, fp)
    
save_name = "MinMax_Scaler.p"   
with open(save_name, "wb") as fp:   #Pickling  
    pickle.dump(mnmx, fp)

save_name = "idx.p"   
with open(save_name, "wb") as fp:   #Pickling  
    pickle.dump(idx, fp)
    
# use some defined model and train it with the x-featues:
model.fit(x_train_transformed, y_train)

files = GoToFiles(save_path)    
# save trained model:    
save_name = model_name +"_Trained"+ ".p"   
with open(save_name, "wb") as fp:   #Pickling  
    pickle.dump(model, fp)

