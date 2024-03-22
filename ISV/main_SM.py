import shap  # https://github.com/slundberg/shap
#import shapreg  # https://github.com/iancovert/shapley-regression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
pd.set_option('expand_frame_repr', False)
np.set_printoptions(threshold=np.inf)
import torch
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data import RandomSampler, BatchSampler
from copy import deepcopy
from tqdm.auto import tqdm
import torch.nn as nn
import itertools
from torch.distributions.categorical import Categorical
import pickle
import os.path
import lightgbm as lgb
import operator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
import csv
import logging
from os.path import isdir, join, exists
from os import rename, remove
from glob import glob
import subprocess
import secrets
from joblib import Parallel, delayed
import time
from operator import itemgetter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels, check_classification_targets
import warnings
from os import chdir, getcwd, mkdir
import shutil
import collections
import matplotlib.patches as mpatches
import math
#from fastshap.utils import ShapleySampler, DatasetRepeat
from tqdm.auto import tqdm
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import t
import random as rn
rn.seed(29)
np.random.seed(29)
torch.manual_seed(29)
torch.backends.cudnn.benchmark = False
#torch.use_deterministic_algorithms(True)
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import chain, combinations
import _pickle as cPickle
import bz2,json,contextlib
import sys

sys.path.append('IntervalSV/code')

from datasets import Monks, Census, Magic, Wbcd, Heart, Diabetes
from original_model import OriginalModel, OriginalModelVV
from utils import *
from surrogate_scaled import Surrogate_Scaled
from adapter import DictionaryData
from gok import MultiFastSHAP
from moore_SM import FastSHAP_M
from han_SM import FastSHAP_U
# from shapreg import PredictionGame, ShapleyRegression
from shapreg_SM import ShapleyRegression2, PredictionGame

import argparse



# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-dataset", help = "Dataset", required=True)
parser.add_argument("-seed", help = "Seed", required=True)
args = parser.parse_args()
dset=args.dataset
SEED=int(args.seed)
dset=dset.lower()
print(dset, SEED)


# Load the config fileù
# json_file = open("config/config.json")
with open(f"config/{dset}_SM.json") as json_file:
    config = json.load(json_file)

lr_mf = float(config["lr_mf"])
lr_m = float(config["lr_m"])
lr_u = float(config["lr_u"])
ki = int(config["ki"])


print("ARGS:", dset, SEED, lr_m, lr_mf, lr_u, ki)


## LOAD DATASET
# SEED=291297
print("\nLoading dataset")
if dset=="magic":
    X, Y, X_test, Y_test, feature_names, dataset = Magic().get_data()
    print(dataset, feature_names)
elif dset=="wbcd":
    X, Y, X_test, Y_test, feature_names, dataset = Wbcd().get_data()
    print(dataset, feature_names)
elif dset=="heart":
    X, Y, X_test, Y_test, feature_names, dataset = Heart().get_data()
    print(dataset, feature_names)
elif dset=="diabetes":
    X, Y, X_test, Y_test, feature_names, dataset = Diabetes().get_data()
    print(dataset, feature_names)
else:
    print("Dataset not found")
    exit()

if Y_test is None: # Census
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=7)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)
else:
    print(X.shape, Y.shape, X_test.shape, Y_test.shape, feature_names, dataset)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3, random_state=0)
    
print(X_train.shape, X_val.shape, X_test.shape)

# Data scaling
num_features = len(feature_names)
ss = StandardScaler()
ss.fit(X_train)
X_train_s = ss.transform(X_train)
X_val_s = ss.transform(X_val)
X_test_s = ss.transform(X_test)

## ORIGINAL MODEL
print("\nTraining original model")
modelRF = RandomForestClassifier(random_state=SEED)
modelRF.fit(X_train_s, Y_train)
om_VV=OriginalModelVV(modelRF)
om=OriginalModel(modelRF)
y_pred=modelRF.predict(X_test_s)
print("Accuracy: ", accuracy_score(Y_test, y_pred))



## FUNCTIONS
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class MultiTaskModel(nn.Module):
    def __init__(self, Layer_size):
        super(MultiTaskModel,self).__init__()
        self.body = nn.Sequential(
            MaskLayer1d(value=0, append=True),
            nn.Linear(2 * num_features, Layer_size), #FA IL CAT DELLA MASK (SUBSET S)
            nn.ReLU(inplace=True),
            nn.Linear(Layer_size, Layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(Layer_size, Layer_size),
            nn.ReLU(inplace=True),
        )
        self.head1 = nn.Sequential(
            nn.Linear(Layer_size, 2),
        )
        self.head2 = nn.Sequential(
            nn.Linear(Layer_size, 2),
        )

    def forward(self,x):
        x = self.body(x)
        v1 = self.head1(x)
        v2 = self.head2(x)
        return v1, v2
    
def original_model_VV(x):
    pred1, pred2 = om_VV.get_pred_VV(x.cpu().numpy()) #MODELLO ORIGINALE, PRED ALWAYS ON POSITIVE CLASS
    return torch.tensor(pred1, dtype=torch.float32, device=x.device), torch.tensor(pred2, dtype=torch.float32, device=x.device)

## SURROGATE SCALED
print("\nTraining surrogate model")
device = torch.device('cpu')
if os.path.isfile(f'models/{dataset}_surrogate_scaled.pt'):
    try:
        print('Loading saved surrogate model')
        surr_VV = torch.load(f'models/{dataset}_surrogate_scaled.pt').to(device)
        surrogate_VV_SM = Surrogate_Scaled(surr_VV, num_features)
    except:
        print("Model saved as dict")
        surr_VV=MultiTaskModel(512)
        surr_VV.load_state_dict(torch.load(f'models/{dataset}_surrogate_scaled.pt'))
        surrogate_VV_SM = Surrogate_Scaled(surr_VV, num_features)
else:
    print("Surrogate not found")
    exit()


## ADAPT SURROGATE OUTPUT
print("\nAdapting surrogate output")

def rearrange_vectors(input_vector):
    vector_1=np.array([input_vector[0][0], input_vector[1][1]])
    vector_2=np.array([input_vector[1][0], input_vector[0][1]])
    return vector_1, vector_2

all_subsets=list(powerset(np.arange(num_features)))#[:-1]

def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)

def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

Flag=False
try:
    train_data_dict=decompress_pickle(f'adapted/{dataset}_train_data_dict.pbz2')
    val_data_dict=decompress_pickle(f'adapted/{dataset}_val_data_dict.pbz2')
except:
    print("\tFirst method load vectors failed")
    Flag=True

if Flag:
    try:
        # load train_data_dict
        with open(f'adapted/{dataset}_train_data_dict.pbz2', 'rb') as f:
            train_data_dict = cPickle.load(f)
        # load val_data_dict
        with open(f'adapted/{dataset}_val_data_dict.pbz2', 'rb') as f:
            val_data_dict = cPickle.load(f)
    except:
        print("\tSecond method load vectors failed")
        exit()



## TRAIN MULTI-FASTSHAP GOK
print("\nTraining FastSHAP MEDIAN")

class MultiTaskExplainer(nn.Module):
    def __init__(self, Layer_size):
        super(MultiTaskExplainer,self).__init__()
        self.body = nn.Sequential(
            nn.Linear(num_features, Layer_size), #FA IL CAT DELLA MASK (SUBSET S)
            nn.LeakyReLU(inplace=True),
            nn.Linear(Layer_size, Layer_size),
            nn.LeakyReLU(inplace=True),
        )
        self.head1 = nn.Sequential(
            nn.Linear(Layer_size, 2*num_features),
        )
        self.head2 = nn.Sequential(
            nn.Linear(Layer_size, 2*num_features),
        )

    def forward(self, x):
        x = self.body(x)
        v1 = self.head1(x)
        v2 = self.head2(x)
        return v1, v2

if os.path.isfile(f'models/{dataset}_explainer_median_seed={SEED}_lr={lr_m}.pt'):
    print('Loading saved Median explainer model')
    explainer_M = torch.load(f'models/{dataset}_explainer_median_seed={SEED}_lr={lr_m}.pt').to(device)
    fastshap_M = FastSHAP_M(explainer_M, surrogate_VV_SM, normalization='additive',link=nn.Softmax(dim=-1))
else:
    print('Training Median explainer model')
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    LAYER_SIZE=512
    explainer_M = nn.Sequential(
        nn.Linear(num_features, LAYER_SIZE),
        nn.LeakyReLU(inplace=True),
        nn.Linear(LAYER_SIZE, LAYER_SIZE),
        nn.LeakyReLU(inplace=True),
        nn.Linear(LAYER_SIZE, 2 * num_features)).to(device)

    # Set up FastSHAP object
    fastshap_M = FastSHAP_M(explainer_M, surrogate_VV_SM, normalization="additive", link=nn.Softmax(dim=-1))

    fastshap_M.train(
        X_train_s,
        X_val_s,
        batch_size=8,
        num_samples=8, 
        max_epochs=400,#200
        validation_samples=128,
        verbose=False,
        paired_sampling=True,
        lr=lr_m, #1e-2
        min_lr=1e-8, #1e-5
        lr_factor=0.5,
        weight_decay=0.05,
        training_seed=SEED,
        lookback=20,
        train_dict_data=train_data_dict,
        val_dict_data=val_data_dict,
    ) ########################################à
    
    explainer_M.cpu()
    torch.save(explainer_M, f'models/{dataset}_explainer_median_seed={SEED}_lr={lr_m}.pt')
    explainer_M.to(device)


print("\nTraining FastSHAP US")
if os.path.isfile(f'models/{dataset}_explainer_us_seed={SEED}_lr={lr_u}.pt'):
    print('Loading saved US explainer model')
    explainer_U = torch.load(f'models/{dataset}_explainer_us_seed={SEED}_lr={lr_u}.pt').to(device)
    fastshap_U = FastSHAP_U(explainer_U, surrogate_VV_SM, normalization='additive',link=nn.Softmax(dim=-1))
else:
    print('Training US explainer model')
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    LAYER_SIZE=512
    explainer_U = nn.Sequential(
        nn.Linear(num_features, LAYER_SIZE),
        nn.LeakyReLU(inplace=True),
        nn.Linear(LAYER_SIZE, LAYER_SIZE),
        nn.LeakyReLU(inplace=True),
        nn.Linear(LAYER_SIZE, 2 * num_features)).to(device)

    # Set up FastSHAP object
    fastshap_U = FastSHAP_U(explainer_U, surrogate_VV_SM, normalization="additive", link=nn.Softmax(dim=-1))

    fastshap_U.train(
        X_train_s,
        X_val_s,
        batch_size=8,
        num_samples=8, 
        max_epochs=400,#200
        validation_samples=128,
        verbose=False,
        paired_sampling=True,
        lr=lr_u, #1e-2
        min_lr=1e-8, #1e-5
        lr_factor=0.5,
        weight_decay=0.05,
        training_seed=SEED,
        lookback=20,
        train_dict_data=train_data_dict,
        val_dict_data=val_data_dict,
    ) ########################################à
    
    explainer_U.cpu()
    torch.save(explainer_U, f'models/{dataset}_explainer_us_seed={SEED}_lr={lr_u}.pt')
    explainer_U.to(device)
    

## LOAD EVALUATE FUNCTIONS
# Setup for KernelSHAP
def get_grand_null_u(x_t, N, y, dict_data):
    
    grand_key="1"*num_features
    DATA_KEY="".join(str(val) for val in x_t[0].data.numpy())
    v=dict_data[DATA_KEY][1][grand_key]
    v1, v2 = rearrange_vectors(v)
    
    if y==0:
        v_u = np.array([-(v2[1]-v1[0])/(2), (v2[1]-v1[0])/(2)])
    else:
        v_u = np.array([-(v1[1]-v2[0])/(2), (v1[1]-v2[0])/(2)])
        

    null_key="0"*num_features
    DATA_KEY="".join(str(val) for val in x_t[0].data.numpy())
    v=dict_data[DATA_KEY][1][null_key]
    v1, v2 = rearrange_vectors(v)
    if y==0:
        v_u_null = np.array([-(v2[1]-v1[0])/(2), (v2[1]-v1[0])/(2)])
    else:
        v_u_null = np.array([-(v1[1]-v2[0])/(2), (v1[1]-v2[0])/(2)])
    diff_u = np.array([v_u[0] - v_u_null[1], v_u[1] - v_u_null[0]])
    
    return v_u, v_u_null, diff_u

# Setup for KernelSHAP
def imputer_lower(x, S):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    S = torch.tensor(S, dtype=torch.float32, device=device)
    # print(S.shape)
    # print(x.shape)
    tmp=[]
    for el, mask in zip(x, S):
        mask_key="".join(str(int(val)) for val in mask.data.numpy())
        DATA_KEY="".join(str(val) for val in el.data.numpy())
        v=train_data_dict[DATA_KEY][1][mask_key]
        v1, v2 = rearrange_vectors(v)
        tmp.append([v1[0],v2[0]])

    #print(tmp)
    mean=np.array(tmp)
    # print(mean.shape)
    mean=torch.as_tensor(mean)
    #mean=mean.softmax(dim=-1)
    return mean.cpu().data.numpy()

# Setup for KernelSHAP
def imputer_upper(x, S):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    S = torch.tensor(S, dtype=torch.float32, device=device)
    tmp=[]
    for el, mask in zip(x, S):
        mask_key="".join(str(int(val)) for val in mask.data.numpy())
        DATA_KEY="".join(str(val) for val in el.data.numpy())
        v=train_data_dict[DATA_KEY][1][mask_key]
        v1, v2 = rearrange_vectors(v)
        tmp.append([v2[1], v1[1]])

    #print(tmp)
    mean=np.array(tmp)
    mean=torch.as_tensor(mean)
    #mean=mean.softmax(dim=-1)
    return mean.cpu().data.numpy()

def ref_phi(median_phi, x, y, N, diff_u):
    interval=[]
    for el in median_phi[:,y]:
        tmp = np.array([el,el])
        tmp = tmp + diff_u/N # NORMALIZED
        interval.append(tmp)
    interval=np.array(interval)
    return interval

def ref_phi_U(median_phi, x, y, N, diff_u):
    interval=[]
    for idx,el in enumerate(median_phi[:,y]):
        tmp = np.array([el,el])
        normaliz=N[idx]/np.sum(N)
        tmp = tmp + diff_u*normaliz # NORMALIZED
        interval.append(tmp)
    interval=np.array(interval)
    return interval

# define a function to compute the euclidean distance between two intervals vectors
def euclidean_distance(interval1, interval2):
    return np.sqrt(np.sum((interval1 - interval2)**2))

print("")
# INDEXES WITH UKS INFERENCE TIME ACCEPTABLE
good_indexes_diabetes=[0,2,5,6,7,11,14,19,20,21,23,24,30,37,39,40,42,44,46,47,49,51,52,54,56,58,60,62,63,64,66,71,72,76,79,81,82,84,86,91,95,97,98,100,101,103,107,109,114,116,118,122,127,131,135,136,139,140,147,148,157,158,165,166,168,171,173,174,176,178,179,180,183,193,198,199,202,206,207,210,213,216,217,219,220,227,230,232,233,236,237,239,241,242,243,244,247,249,250,251]
good_indexes_heart=np.arange(0,100) #primi 100
good_indexes_wbcd=[0,3,4,5,14,15,19,20,26,28,29,30,32,33,36,40,43,44,46,48,49,51,54,55,56,57,63,64,65,67,68,69,70,73,74,77,81,82,83,87,88,91,93,94,95,96,106,107,108,109,111,114,116,117,118,119,120,122,123,124,125,126,135,137,138,139,141,143,147,150,151,153,156,159,164,165,169,174,175,179,180,181,182,184,186,190,199,207,208,210,212,218,222,224,227,228,229,230,235,236]
good_indexes_magic=[4,5,7,16,20,21,24,27,28,29,31,34,37,38,40,41,43,45,49,50,51,53,54,57,59,64,65,70,75,79,82,84,85,88,89,90,91,95,102,103,104,109,112,115,116,119,120,124,126,127,129,130,132,133,139,141,142,147,149,152,153,155,157,158,159,160,163,164,167,169,171,173,175,176,178,181,182,184,187,188,191,192,193,195,196,199,200,203,204,208,209,212,220,221,223,224,227,229,235,236]
if dataset=="Magic":
    print("Magic indexes")
    good_indexes=good_indexes_magic
if dataset=="WBCD":
    print("WBCD indexes")
    good_indexes=good_indexes_wbcd
if dataset=="Diabetes":
    print("Diabetes indexes")
    good_indexes=good_indexes_diabetes
if dataset=="Heart":
    print("Heart indexes")
    good_indexes=good_indexes_heart

## TUNING ALPHA
print("\nTuning alpha")
device = torch.device('cpu')

for alpha in [0, 0.001, 0.005, 0.01, 0.05, 0.1]:
# for alpha in [0.99]:
    print("#"*100)
    print(f"Alpha: {alpha}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if os.path.isfile(f"models/{dataset}_multiexplainer_alpha_{alpha}_SEED={SEED}_lr={lr_mf}.pt"): #####################################################
        print('Loading saved MF explainer model')
        explainer3 = torch.load(f"models/{dataset}_multiexplainer_alpha_{alpha}_SEED={SEED}_lr={lr_mf}.pt").to(device)
        fastshap3 = MultiFastSHAP(explainer3, surrogate_VV_SM, normalization='additive', link=nn.Softmax(dim=-1))
    else:
        print("Training new MF explainer model")
        explainer3 = MultiTaskExplainer(512).to(device)
        fastshap3 = MultiFastSHAP(explainer3, surrogate_VV_SM, normalization="additive", link=nn.Softmax(dim=-1))

        # Train
        fastshap3.train(
            X_train_s,
            X_val_s,
            batch_size=8,
            num_samples=8, 
            max_epochs=400,
            validation_samples=128,
            verbose=False,
            paired_sampling=True,
            approx_null=True,
            lr=lr_mf,
            min_lr=1e-8,
            lr_factor=0.5,
            weight_decay=0.05,
            training_seed=SEED,
            lookback=20,
            debug=False,
            debug_val=False,
            constraint=True,    ########################################
            alpha=alpha,        ########################################
            train_dict_data=train_data_dict,
            val_dict_data=val_data_dict,
        )  
        
        # save the model
        torch.save(explainer3, f"models/{dataset}_multiexplainer_alpha_{alpha}_SEED={SEED}_lr={lr_mf}.pt")

    mean_error_rate_mfs = []
    mean_error_rate_ts = []
    mean_error_rate_ks = []
    mean_error_rate_ms = []
    mean_error_rate_hs = []


    L2_distances_tmf = []
    L2_distances_tk = []
    L2_distances_tm = []
    L2_distances_th = []


    L2_distances_tmf_err = []
    L2_distances_tk_err = []
    L2_distances_tm_err = []
    L2_distances_th_err = []


    L1_distances_tmf = []
    L1_distances_tk = []
    L1_distances_tm = []
    L1_distances_th = []


    L1_distances_tmf_err = []
    L1_distances_tk_err = []
    L1_distances_tm_err = []
    L1_distances_th_err = []

    # if alpha==0:
    average_ts = []
    average_ks = []
    average_ts_err = []
    average_ks_err = []
    list_uks=[]
    list_ks=[]

    average_mfs = []
    average_ms = []
    average_hs = []

    average_mfs_err = []
    average_ms_err = []
    average_hs_err = []
    
    Eucl_UKS_GOK=[]
    Eucl_UKS_KS=[]
    Eucl_UKS_F=[]
    Eucl_UKS_H=[]
    

    
    iter=0
    for ind in tqdm(good_indexes[:]):
        x = X_train_s[ind:ind+1]
        y = int(Y_train[ind])
        
        game_l = PredictionGame(imputer_lower, x)
        game_u = PredictionGame(imputer_upper, x)
        shap_values_l, shap_values_u, all_results = ShapleyRegression2(game_1=game_l, game_2=game_u, batch_size=64, num_features=num_features, detect_convergence=True, bar=False, thresh=0.3, return_all=True)
        
        # if alpha==0:
        ##########################
        #   UnbiasedKernelSHAP   #
        ##########################
        tmp1 = shap_values_l.values[:, y]
        tmp2 = shap_values_u.values[:, y]
        error_rate_ts = 0
        isv_uks=[]
        for el1, el2 in zip(tmp1, tmp2):
            # print(el1,el2)
            isv_uks.append([el1, el2])
            if el1 > el2:
                error_rate_ts += 1
        mean_ts = (tmp2 + tmp1) / 2
        err_ts = np.abs(mean_ts - tmp2)
        mean_error_rate_ts.append(error_rate_ts)
        average_ts.append(mean_ts)
        average_ts_err.append(err_ts)
        
        isv_uks=np.array(isv_uks)
        list_uks.append(isv_uks)
        
        
        ##########################
        #    BiasedKernelSHAP    #
        ##########################
        kernelshap_iters = ki
        tmp1 = all_results['values_1'][list(all_results['iters']).index(kernelshap_iters)][:, y]
        tmp2 = all_results['values_2'][list(all_results['iters']).index(kernelshap_iters)][:, y]
        error_rate_ks = 0
        isv_ks=[]
        for el1, el2 in zip(tmp1, tmp2):
            # print(el1,el2)
            isv_ks.append([el1, el2])
            if el1 > el2:
                error_rate_ks += 1

        mean_ks = (tmp2 + tmp1) / 2
        err_ks = np.abs(mean_ks - tmp2)
        mean_error_rate_ks.append(error_rate_ks)
        average_ks.append(mean_ks)
        average_ks_err.append(err_ks)
        
        isv_ks=np.array(isv_ks)
        list_ks.append(isv_ks)
        
        ##########################
        #   Gok MultiFastSHAP    #
        ##########################
        multi1, multi2 = fastshap3.shap_values(x, train_data_dict, False)
        multi1 = multi1[0, :, :]
        multi2 = multi2[0, :, :]
        
        multifastshap_values_mean = []
        multifastshap_values_ci = []
        error_rate_mfs = 0
        isv_mf=[]
        for el1, el2 in zip(multi1, multi2):
            if y == 0:
                isv_mf.append([el1[0], el2[1]])
                if el1[0] > el2[1]:
                    error_rate_mfs += 1
            else:
                isv_mf.append([el2[0], el1[1]])
                if el2[0] > el1[1]:
                    error_rate_mfs += 1
            m1 = (el1[0] + el2[1]) / 2
            m2 = (el2[0] + el1[1]) / 2
            c1 = np.abs(m1 - el1[0])
            c2 = np.abs(m2 - el2[0])

            multifastshap_values_mean.append([m1, m2])
            multifastshap_values_ci.append([c1, c2])
        mean_error_rate_mfs.append(error_rate_mfs)
        multifastshap_values_mean = np.array(multifastshap_values_mean)
        multifastshap_values_ci = np.array(multifastshap_values_ci)
        mean_ms=multifastshap_values_mean[:,y]
        err_ms=multifastshap_values_ci[:,y]
        average_mfs.append(mean_ms)
        average_mfs_err.append(err_ms)
        
        isv_mf=np.array(isv_mf)
        
        ##########################
        #   FENG MultiFastSHAP   #
        ##########################
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        grand_u, null_u, diff_u = get_grand_null_u(x_t, num_features, y, train_data_dict)
        isv_moore=fastshap_M.shap_values(x_t, train_data_dict, False)[0]
        isv_0 = ref_phi(isv_moore, x, y, num_features, diff_u)
        error_rate_feng=0
        for el in isv_0:
            if el[0] > el[1]:
                error_rate_feng += 1
        mean_feng=np.mean(isv_0, axis=1)
        err_feng=np.abs(isv_0[:,1]-mean_feng)
        mean_error_rate_ms.append(error_rate_feng)
        average_ms.append(mean_feng)
        average_ms_err.append(err_feng)
        
        isv_feng=isv_0
        
        ##########################
        #    HAN MultiFastSHAP   #
        ##########################
        isv_han=fastshap_U.shap_values(x_t, train_data_dict, False)[0]
        arr_phi_u=isv_han[:,y]
        width_phi_u = arr_phi_u*2
        # print("HAN FS")
        isv_3 = ref_phi_U(isv_moore, x, y, width_phi_u, diff_u)
        error_rate_han=0
        for el in isv_3:
            if el[0] > el[1]:
                error_rate_han += 1
        mean_han=np.mean(isv_3, axis=1)
        err_han=isv_3[:,1]-mean_han
        mean_error_rate_hs.append(error_rate_han)
        average_hs.append(mean_han)
        average_hs_err.append(err_han)
        
        isv_han=isv_3
        
        ##########################
        #         METRICS        #
        ##########################
        # Calculate L2 distances
        L2_distances_tmf.append(np.linalg.norm(mean_ms - average_ts[iter]))
        L2_distances_tk.append(np.linalg.norm(average_ks[iter] - average_ts[iter]))
        L2_distances_tm.append(np.linalg.norm(mean_feng - average_ts[iter]))
        L2_distances_th.append(np.linalg.norm(mean_han - average_ts[iter]))

        L2_distances_tmf_err.append(np.linalg.norm(err_ms - average_ts_err[iter]))
        L2_distances_tk_err.append(np.linalg.norm(average_ks_err[iter] - average_ts_err[iter]))
        L2_distances_tm_err.append(np.linalg.norm(err_feng - average_ts_err[iter]))
        L2_distances_th_err.append(np.linalg.norm(err_han - average_ts_err[iter]))

        # Compute L1 distances
        L1_distances_tmf.append(np.linalg.norm(mean_ms - average_ts[iter], ord=1))
        L1_distances_tk.append(np.linalg.norm(average_ks[iter] - average_ts[iter], ord=1))
        L1_distances_tm.append(np.linalg.norm(mean_feng - average_ts[iter], ord=1))
        L1_distances_th.append(np.linalg.norm(mean_han - average_ts[iter], ord=1))

        L1_distances_tmf_err.append(np.linalg.norm(err_ms - average_ts_err[iter], ord=1))
        L1_distances_tk_err.append(np.linalg.norm(average_ks_err[iter] - average_ts_err[iter], ord=1))
        L1_distances_tm_err.append(np.linalg.norm(err_feng - average_ts_err[iter], ord=1))
        L1_distances_th_err.append(np.linalg.norm(err_han - average_ts_err[iter], ord=1))
        
        Eucl_UKS_GOK.append(euclidean_distance(list_uks[iter], isv_mf))
        Eucl_UKS_KS.append(euclidean_distance(list_uks[iter], list_ks[iter]))
        Eucl_UKS_F.append(euclidean_distance(list_uks[iter], isv_feng))
        Eucl_UKS_H.append(euclidean_distance(list_uks[iter], isv_han))
        
        iter+=1
        

    # Print the error rates
    print("\tError rates:")
    print("\tUnbiasedKernelSHAP:", np.mean(mean_error_rate_ts))
    print("\tMulti-FastSHAP-GOK:", np.mean(mean_error_rate_mfs))
    print("\tMulti-FastSHAP-FENG:", np.mean(mean_error_rate_ms))
    print("\tMulti-FastSHAP-HAN:", np.mean(mean_error_rate_hs))
    print("\tKernelSHAP:", np.mean(Eucl_UKS_KS))

    print("")
    print("\t","-"*100)
    print("")
    
    print("\tEuclidean Distances:")
    print("\tUKS-GOK:", np.mean(Eucl_UKS_GOK))
    print("\tUKS-FENG:", np.mean(Eucl_UKS_F))
    print("\tUKS-HAN:", np.mean(Eucl_UKS_H))
    print("\tUKS-KS", np.mean(Eucl_UKS_KS))

    print("")
    print("\t","-"*100)
    print("")

    # Print the L2 distances
    print("\tL2 distances:")
    print("\tMulti-FastSHAP-GOK:", np.mean(L2_distances_tmf))
    print("\tMulti-FastSHAP-FENG:", np.mean(L2_distances_tm))
    print("\tMulti-FastSHAP-HAN:", np.mean(L2_distances_th))
    print("\tKernelSHAP:", np.mean(L2_distances_tk))

    print("")

    # Print the L2 distances on err
    print("\tL2 distances on err:")
    print("\tMulti-FastSHAP-GOK:", np.mean(L2_distances_tmf_err))
    print("\tMulti-FastSHAP-FENG:", np.mean(L2_distances_tm_err))
    print("\tMulti-FastSHAP-HAN:", np.mean(L2_distances_th_err))
    print("\tKernelSHAP:", np.mean(L2_distances_tk_err))

    print("")
    print("\t","-"*100)
    print("")

    # Print the L1 distances
    print("\tL1 distances:")
    print("\tMulti-FastSHAP-GOK:", np.mean(L1_distances_tmf))
    print("\tMulti-FastSHAP-FENG:", np.mean(L1_distances_tm))
    print("\tMulti-FastSHAP-HAN:", np.mean(L1_distances_th))
    print("\tKernelSHAP:", np.mean(L1_distances_tk))

    print("")

    # Print the L1 distances on err
    print("\tL1 distances on err:")
    print("\tMulti-FastSHAP-GOK:", np.mean(L1_distances_tmf_err))
    print("\tMulti-FastSHAP-FENG:", np.mean(L1_distances_tm_err))
    print("\tMulti-FastSHAP-HAN:", np.mean(L1_distances_th_err))
    print("\tKernelSHAP:", np.mean(L1_distances_tk_err))

    with open(f"results/{dataset}_results_alpha={alpha}_SEED={SEED}.txt", "a+") as file:
        file.write("#"*100)
        file.write(f"Alpha: {alpha}\n")
        file.write("\tError rates:\n")
        file.write(f"\tUnbiasedKernelSHAP: {np.mean(mean_error_rate_ts)}\n")
        file.write(f"\tMulti-FastSHAP-GOK: {np.mean(mean_error_rate_mfs)}\n")
        file.write(f"\tMulti-FastSHAP-FENG: {np.mean(mean_error_rate_ms)}\n")
        file.write(f"\tMulti-FastSHAP-HAN: {np.mean(mean_error_rate_hs)}\n")
        file.write(f"\tKernelSHAP: {np.mean(mean_error_rate_ks)}\n")
        # file.write("\n")
        file.write("-"*100)
        # file.write("\n")
        file.write("\tEuclidean Distances:\n")
        file.write(f"\tUKS-GOK: {np.mean(Eucl_UKS_GOK)}\n")
        file.write(f"\tUKS-FENG: {np.mean(Eucl_UKS_F)}\n")
        file.write(f"\tUKS-HAN: {np.mean(Eucl_UKS_H)}\n")
        file.write(f"\tUKS-KS: {np.mean(Eucl_UKS_KS)}\n")
        # file.write("\n")
        file.write("-"*100)
        # # file.write("\n")
        file.write("\tL2 distances:\n")
        file.write(f"\tMulti-FastSHAP-GOK: {np.mean(L2_distances_tmf)}\n")
        file.write(f"\tMulti-FastSHAP-FENG: {np.mean(L2_distances_tm)}\n")
        file.write(f"\tMulti-FastSHAP-HAN: {np.mean(L2_distances_th)}\n")
        file.write(f"\tKernelSHAP: {np.mean(L2_distances_tk)}\n")
        # file.write("\n")
        file.write("\tL2 distances on err:\n")
        file.write(f"\tMulti-FastSHAP-GOK: {np.mean(L2_distances_tmf_err)}\n")
        file.write(f"\tMulti-FastSHAP-FENG: {np.mean(L2_distances_tm_err)}\n")
        file.write(f"\tMulti-FastSHAP-HAN: {np.mean(L2_distances_th_err)}\n")
        file.write(f"\tKernelSHAP: {np.mean(L2_distances_tk_err)}\n")
        # file.write("\n")
        file.write("-"*100)
        # file.write("\n")
        file.write("\tL1 distances:\n")
        file.write(f"\tMulti-FastSHAP-GOK: {np.mean(L1_distances_tmf)}\n")
        file.write(f"\tMulti-FastSHAP-FENG: {np.mean(L1_distances_tm)}\n")
        file.write(f"\tMulti-FastSHAP-HAN: {np.mean(L1_distances_th)}\n")
        file.write(f"\tKernelSHAP: {np.mean(L1_distances_tk)}\n")
        # file.write("\n")
        file.write("\tL1 distances on err:\n")
        file.write(f"\tMulti-FastSHAP-GOK: {np.mean(L1_distances_tmf_err)}\n")
        file.write(f"\tMulti-FastSHAP-FENG: {np.mean(L1_distances_tm_err)}\n")
        file.write(f"\tMulti-FastSHAP-HAN: {np.mean(L1_distances_th_err)}\n")
        file.write(f"\tKernelSHAP: {np.mean(L1_distances_tk_err)}\n")

    SAVE=True
    if SAVE:
        # save all the metrics in a pickle file inside the folder 'dump'
        with open(f"dump/{dataset}_results_alpha={alpha}_SEED={SEED}.pkl", "wb") as file:
            pickle.dump({
                "mean_error_rate_mfs": mean_error_rate_mfs,
                "mean_error_rate_ts": mean_error_rate_ts,
                "mean_error_rate_ks": mean_error_rate_ks,
                "mean_error_rate_ms": mean_error_rate_ms,
                "mean_error_rate_hs": mean_error_rate_hs,
                "L2_distances_tmf": L2_distances_tmf,
                "L2_distances_tk": L2_distances_tk,
                "L2_distances_tm": L2_distances_tm,
                "L2_distances_th": L2_distances_th,
                "L2_distances_tmf_err": L2_distances_tmf_err,
                "L2_distances_tk_err": L2_distances_tk_err,
                "L2_distances_tm_err": L2_distances_tm_err,
                "L2_distances_th_err": L2_distances_th_err,
                "L1_distances_tmf": L1_distances_tmf,
                "L1_distances_tk": L1_distances_tk,
                "L1_distances_tm": L1_distances_tm,
                "L1_distances_th": L1_distances_th,
                "L1_distances_tmf_err": L1_distances_tmf_err,
                "L1_distances_tk_err": L1_distances_tk_err,
                "L1_distances_tm_err": L1_distances_tm_err,
                "L1_distances_th_err": L1_distances_th_err,
                "average_ts": average_ts,
                "average_ks": average_ks,
                "average_mfs": average_mfs,
                "average_ms": average_ms,
                "average_hs": average_hs,
                "average_ts_err": average_ts_err,
                "average_ks_err": average_ks_err,
                "average_mfs_err": average_mfs_err,
                "average_ms_err": average_ms_err,
                "average_hs_err": average_hs_err,
                "Eucl_UKS_GOK": Eucl_UKS_GOK,
                "Eucl_UKS_KS": Eucl_UKS_KS,
                "Eucl_UKS_F": Eucl_UKS_F,
                "Eucl_UKS_H": Eucl_UKS_H
            }, file)