import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

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
# import lightgbm as lgb
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
# import matplotlib.patches as mpatches
import math
#from fastshap.utils import ShapleySampler, DatasetRepeat
from tqdm.auto import tqdm
import random
import scipy.stats as stats
# import matplotlib.pyplot as plt
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

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import sys

sys.path.append('IntervalSV/code')

from datasets import Monks, Census, Magic, Wbcd, Heart, Diabetes, Bank, Credit, Mozilla, Phoneme
from original_model import OriginalModel, OriginalModelVV
from utils import *
from surrogate import Surrogate_VV
from median_LIKE import FastSHAP_M
from us_LIKE import FastSHAP_U
# from shapreg import PredictionGame, ShapleyRegression
from shapreg import ShapleyRegression, PredictionGame
from shapreg_LIKE import ShapleyRegression_U

import argparse



# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-dataset", help = "Dataset", required=True)
parser.add_argument("-seed", help = "Seed", required=True)
args = parser.parse_args()
dset=args.dataset
dset=dset.lower()
SEED=int(args.seed)


# Load the config file
with open(f"config_LIKE/{dset.lower()}.json") as json_file:
    config = json.load(json_file)


lr_m = float(config["lr_m"])
lr_u = float(config["lr_u"])
ki = int(config["ki"])
M=int(config["M"])
BS_UKS=int(config["BS_UKS"])
TH=float(config["TH"])

# Read arguments from command line


# lr_m=float(args.lr_m)
# lr_u=float(args.lr_u)
# lr_mf=float(args.lr_mf)
# ki=int(args.ki)

print("ARGS:", dset, SEED, lr_m, lr_u, ki, M, BS_UKS, TH)


## LOAD DATASET
# SEED=291297
print("\nLoading dataset")
if dset=="magic":
    X, Y, X_test, Y_test, feature_names, dataset = Magic().get_data()
elif dset=="wbcd":
    X, Y, X_test, Y_test, feature_names, dataset = Wbcd().get_data()
elif dset=="heart":
    X, Y, X_test, Y_test, feature_names, dataset = Heart().get_data()
elif dset=="diabetes":
    X, Y, X_test, Y_test, feature_names, dataset = Diabetes().get_data()
elif dset=="census":
    X, Y, X_test, Y_test, feature_names, dataset = Census().get_data()
elif dset=="monks":
    X, Y, X_test, Y_test, feature_names, dataset = Monks().get_data()
elif dset=="credit":
    X, Y, X_test, Y_test, feature_names, dataset = Credit().get_data()
elif dset=="bank":
    X, Y, X_test, Y_test, feature_names, dataset = Bank().get_data()
elif dset=="mozilla":
    X, Y, X_test, Y_test, feature_names, dataset = Mozilla().get_data()
elif dset=="phoneme":
    X, Y, X_test, Y_test, feature_names, dataset = Phoneme().get_data()


print(dataset, feature_names)


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

## SURROGATE VV
print("\nTraining surrogate model")
device = torch.device('cpu')
if os.path.isfile(f'models_LIKE/{dataset}_surrogate_VV.pt'):
    print('Loading saved surrogate model')
    surr_VV = torch.load(f'models_LIKE/{dataset}_surrogate_VV.pt').to(device)
    surrogate_VV = Surrogate_VV(surr_VV, num_features)
else:
    print("Surrogate not found")
    exit()


link=nn.Softmax(dim=-1)

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

if os.path.isfile(f'models_LIKE/{dataset}_explainer_median_seed={SEED}_LR={lr_m}.pt'):
    print('Loading saved Median explainer model')
    explainer_M = torch.load(f'models_LIKE/{dataset}_explainer_median_seed={SEED}_LR={lr_m}.pt').to(device)
    fastshap_M = FastSHAP_M(explainer_M, surrogate_VV, normalization='additive',link=nn.Softmax(dim=-1))
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
    fastshap_M = FastSHAP_M(explainer_M, surrogate_VV, normalization="additive", link=nn.Softmax(dim=-1))

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
    ) ########################################à
    
    explainer_M.cpu()
    torch.save(explainer_M, f'models_LIKE/{dataset}_explainer_median_seed={SEED}_LR={lr_m}.pt')
    explainer_M.to(device)


print("\nTraining FastSHAP US")
if os.path.isfile(f'models_LIKE/{dataset}_explainer_us_seed={SEED}_LR={lr_u}.pt'):
    print('Loading saved US explainer model')
    explainer_U = torch.load(f'models_LIKE/{dataset}_explainer_us_seed={SEED}_LR={lr_u}.pt').to(device)
    fastshap_U = FastSHAP_U(explainer_U, surrogate_VV, normalization='additive',link=nn.Softmax(dim=-1))
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
    fastshap_U = FastSHAP_U(explainer_U, surrogate_VV, normalization="additive", link=nn.Softmax(dim=-1))

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
    ) ########################################à
    
    explainer_U.cpu()
    torch.save(explainer_U, f'models_LIKE/{dataset}_explainer_us_seed={SEED}_LR={lr_u}.pt')
    explainer_U.to(device)
    

## LOAD EVALUATE FUNCTIONS
# Setup for KernelSHAP# Setup for KernelSHAP
def imputer(x, S):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    S = torch.tensor(S, dtype=torch.float32, device=device)
    pred1, pred2 = surrogate_VV(x, S)#.softmax(dim=-1)
    pred1=pred1.softmax(dim=-1)
    pred2=pred2.softmax(dim=-1)
    tmp1=pred1.detach().numpy()
    tmp2=pred2.detach().numpy()
    # print("-"*100)
    # print(x.shape)
    # print(S.shape)
    tmp=[]
    for index in range(len(tmp1)):
        tmp.append([ (tmp1[index][0]+tmp2[index][1])/2,  (tmp2[index][0]+tmp1[index][1])/2])
    # print(tmp)
    mean=np.array(tmp)
    # print(mean.shape)
    mean=torch.as_tensor(mean)
    
    return mean.cpu().data.numpy()

# Setup for KernelSHAP
def imputer_U(x, S):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    S = torch.tensor(S, dtype=torch.float32, device=device)
    pred1, pred2 = surrogate_VV(x, S)#.softmax(dim=-1)
    pred1=pred1.softmax(dim=-1)
    pred2=pred2.softmax(dim=-1)
    tmp1=pred1.detach().numpy()
    tmp2=pred2.detach().numpy()
    # print("-"*100)
    # print(x.shape)
    # print(S.shape)
    tmp=[]
    for index in range(len(tmp1)):
        tmp.append([ np.abs(tmp1[index][0]-tmp2[index][1])/2,  np.abs(tmp2[index][0]-tmp1[index][1])/2])
    # print(tmp)
    mean=np.array(tmp)
    # print(mean.shape)
    mean=torch.as_tensor(mean)
    
    return mean.cpu().data.numpy()


def MonteCarlo_UNI(IND, DATA, MODEL, OM, SURROGATE, M):

    sv_m_mc=[]
    sv_u_mc=[]

    x=DATA.iloc[IND]
    ones = torch.ones(1, N, dtype=torch.float32)

    for j in range(num_features):
        #M = 750 #####################################################################################################################################################################
        n_features = len(x)
        marginal_contributions = []
        marginal_contributions_U =[]

        feature_idxs = list(range(n_features))
        feature_idxs.remove(j)
        for itr in range(M):
            z = DATA.sample(1).values[0]
            x_idx = random.sample(feature_idxs, min(max(int(0.2*n_features), random.choice(feature_idxs)), int(0.8*n_features))) #estraggo 0.8*feature_idx
            z_idx = [idx for idx in feature_idxs if idx not in x_idx] # features non estratte. Ricorda che una feauture, quella su cui si calcola lo SV, è sempre esclusa

            # construct two new instances
            x_plus_j = np.array([x[i] if i in x_idx + [j] else z[i] for i in range(n_features)])
            x_minus_j = np.array([z[i] if i in z_idx + [j] else x[i] for i in range(n_features)])

            ##############################################################################à
            # calculate marginal contribution
            x_plus_j=x_plus_j.reshape(1, -1)#np.expand_dims(x_plus_j, axis=0)
            x_minus_j=x_minus_j.reshape(1, -1)#np.expand_dims(x_minus_j, axis=0)
            # print(x_plus_j)
            # print(x_minus_j)


            # TODO
            # MARGINAL CONTRIBUTION FROM THE SURROGATE
            v1, v2 = SURROGATE(torch.tensor(x_plus_j, dtype=torch.float32), ones)
            v1 = link(v1[0])
            v2 = link(v2[0])
            v1 = v1.data.numpy()
            v2 = v2.data.numpy()
            # v1, v2 = OM.get_pred_VV(x_plus_j)
            # v1 = v1[0]
            # v2 = v2[0]
            # print(v1, v2)
            v_m_plus = np.array([(v1[0]+v2[1])/2, (v2[0]+v1[1])/2])
            v_m_plus_U = np.array([np.abs(v1[0]-v2[1])/2, np.abs(v2[0]-v1[1])/2])

            v1, v2 = SURROGATE(torch.tensor(x_minus_j, dtype=torch.float32), ones)
            v1 = link(v1[0])
            v2 = link(v2[0])
            v1 = v1.data.numpy()
            v2 = v2.data.numpy()
            # v1, v2 = OM.get_pred_VV(x_minus_j)
            # v1 = v1[0]
            # v2 = v2[0]
            # print(v1, v2)
            v_m_minus = np.array([(v1[0]+v2[1])/2, (v2[0]+v1[1])/2])
            v_m_minus_U = np.array([np.abs(v1[0]-v2[1])/2, np.abs(v2[0]-v1[1])/2])
            
            # print(v_m_plus)
            # print(v_m_minus)

            marginal_contribution = v_m_plus - v_m_minus 
            marginal_contributions.append(marginal_contribution)
            
            marginal_contribution_U = np.abs(v_m_plus_U - v_m_minus_U)   # NO MARGINAL CONTRIBUTUION NEGATIVE?
            marginal_contributions_U.append(marginal_contribution_U)
            # break

        marginal_contributions=np.array(marginal_contributions)
        marginal_contributions_U=np.array(marginal_contributions_U)

        phi_j_x = np.sum(marginal_contributions, axis=0) / len(marginal_contributions)  # our shaply value
        phi_j_x_U = np.sum(marginal_contributions_U, axis=0) / len(marginal_contributions_U)  # our shaply value
        # break

        sv_m_mc.append(phi_j_x)
        sv_u_mc.append(phi_j_x_U)
        
    return np.array(sv_m_mc), np.array(sv_u_mc)


def get_grand_null(x_t, N, y):
    ones = torch.ones(1, N, dtype=torch.float32)
    zeros = torch.zeros(1, N, dtype=torch.float32)
    v1, v2 = surrogate_VV(x_t, ones)
    v1 = link(v1[0])
    v2 = link(v2[0])
    v1 = v1.data.numpy()
    v2 = v2.data.numpy()
    # print(v1, v2)
    if y==0:
        grand_y = np.array([v1[0], v2[1]])
    else:
        grand_y = np.array([v2[0], v1[1]])
    # print(grand_y)
    
    v1, v2 = surrogate_VV(x_t, zeros)
    v1 = link(v1[0])
    v2 = link(v2[0])
    v1 = v1.data.numpy()
    v2 = v2.data.numpy()
    # print(v1, v2)
    if y==0:
        null_y = np.array([v1[0], v2[1]])
    else:
        null_y = np.array([v2[0], v1[1]])
    # print(null_y)
    diff_y = np.array([grand_y[0] - null_y[1], grand_y[1] - null_y[0]])
    return grand_y, null_y, diff_y


def get_grand_null_m(x_t, N, y):
    ones = torch.ones(1, N, dtype=torch.float32)
    zeros = torch.zeros(1, N, dtype=torch.float32)
    v1, v2 = surrogate_VV(x_t, ones)
    v1 = link(v1[0])
    v2 = link(v2[0])
    v1 = v1.data.numpy()
    v2 = v2.data.numpy()
    grand = np.array([(v1[0]+v2[1])/2, (v2[0]+v1[1])/2])
    
    v1, v2 = surrogate_VV(x_t, zeros)
    v1 = link(v1[0])
    v2 = link(v2[0])
    v1 = v1.data.numpy()
    v2 = v2.data.numpy()
    null = np.array([(v1[0]+v2[1])/2, (v2[0]+v1[1])/2])
    diff = np.array([grand[0] - null[1], grand[1] - null[0]])
    return grand, null, diff


def get_grand_null_u(x_t, N, y):
    link=nn.Softmax(dim=-1)
    # x_t=torch.tensor(x, dtype=torch.float32)
    ones = torch.ones(1, N, dtype=torch.float32)
    v1, v2 = surrogate_VV(x_t, ones)
    v1 = link(v1[0])
    v2 = link(v2[0])
    v1 = v1.data.numpy()
    v2 = v2.data.numpy()
    if y==0:
        v_u = np.array([-(v2[1]-v1[0])/(2), (v2[1]-v1[0])/(2)])
    else:
        v_u = np.array([-(v1[1]-v2[0])/(2), (v1[1]-v2[0])/(2)])
        
    zeros = torch.ones(1, N, dtype=torch.float32)
    v1, v2 = surrogate_VV(x_t, zeros)
    v1 = link(v1[0])
    v2 = link(v2[0])
    v1 = v1.data.numpy()
    v2 = v2.data.numpy()
    if y==0:
        v_u_null = np.array([-(v2[1]-v1[0])/(2), (v2[1]-v1[0])/(2)])
    else:
        v_u_null = np.array([-(v1[1]-v2[0])/(2), (v1[1]-v2[0])/(2)])
    diff_u = np.array([v_u[0] - v_u_null[1], v_u[1] - v_u_null[0]])
    return v_u, v_u_null, diff_u


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


print("\nLOOP DATA")

kernelshap_iters = ki
X_train_s_TMP=pd.DataFrame(X_train_s, columns=feature_names)
N=num_features
L1_F_FS=[]
L1_F_KS=[]
L1_F_MC=[]
L1_H_FS=[]
L1_H_KS=[]
L1_H_MC=[]
L1_F_FS_err=[]
L1_F_KS_err=[]
L1_F_MC_err=[]
L1_H_FS_err=[]
L1_H_KS_err=[]
L1_H_MC_err=[]

L2_F_FS=[]
L2_F_KS=[]
L2_F_MC=[]
L2_H_FS=[]
L2_H_KS=[]
L2_H_MC=[]
L2_F_FS_err=[]
L2_F_KS_err=[]
L2_F_MC_err=[]
L2_H_FS_err=[]
L2_H_KS_err=[]
L2_H_MC_err=[]

Eucl_F_UKS_FS=[]
Eucl_F_UKS_MC=[]
Eucl_F_UKS_KS=[]
Eucl_H_UKS_FS=[]
Eucl_H_UKS_MC=[]
Eucl_H_UKS_KS=[]
Eucl_F_H=[]

average_F_UKS=[]
average_F_MC=[]
average_F_KS=[]
average_F_FS=[]
average_H_UKS=[]
average_H_MC=[]
average_H_KS=[]
average_H_FS=[]

error_F_UKS=[]
error_F_MC=[]
error_F_KS=[]
error_F_FS=[]
error_H_UKS=[]
error_H_MC=[]
error_H_KS=[]
error_H_FS=[]

time_F_UKS=[]
time_F_MC=[]
time_F_KS=[]
time_F_FS=[]
time_H_UKS=[]
time_H_MC=[]
time_H_KS=[]
time_H_FS=[]



for ind in tqdm(range(len(X_train_s[:100]))):
    # print(ind)

    x = X_train_s[ind:ind + 1]
    y = int(Y_train[ind])
    
    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    

    
    # print("FENG UKS+KS")
    try:
        grand, null, diff = get_grand_null(x_t, num_features, y)
        grand_m, null_m, diff_m = get_grand_null_m(x_t, num_features, y)
        grand_u, null_u, diff_u = get_grand_null_u(x_t, num_features, y)

        start=time.time()
        fs_m=fastshap_M.shap_values(x)[0]
        time_F_FS.append(time.time()-start)
        # print("FENG FS")
        isv_0 = ref_phi(fs_m, x, y, num_features, diff_u)
        game = PredictionGame(imputer, x)
        start=time.time()
        shap_values, all_results = ShapleyRegression(game, batch_size=BS_UKS, paired_sampling=False, detect_convergence=True, bar=False, return_all=True, thresh=TH) ###########################################################################
        time_F_KS.append(time.time()-start)
        time_F_UKS.append(time.time()-start)
        # print("HAN UKS+KS")
        game_U = PredictionGame(imputer_U, x)
        start=time.time()
        shap_values_u, all_results_u = ShapleyRegression_U(game_U, batch_size=BS_UKS, paired_sampling=False, detect_convergence=True, bar=False, return_all=True, thresh=TH) ###########################################################################
        time_H_KS.append(time.time()-start)
        time_H_UKS.append(time.time()-start)


        isv_1 = ref_phi(shap_values.values, x, y, num_features, diff_u)
        isv_2 = ref_phi(all_results['values'][list(all_results['iters']).index(kernelshap_iters)], x, y, num_features, diff_u)
        
        start=time.time()
        fs_u=fastshap_U.shap_values(x)[0]
        time_H_FS.append((time.time()-start))
        arr_phi_u=fs_u[:,y]
        width_phi_u = arr_phi_u*2
        # print("HAN FS")
        isv_3 = ref_phi_U(fs_m, x, y, width_phi_u, diff_u)
        
        
        
        arr_phi_u2=np.abs(shap_values_u.values[:,y])
        width_phi_u2 = arr_phi_u2*2
        isv_4 = ref_phi_U(shap_values.values, x, y, width_phi_u2, diff_u)

        arr_phi_u3=np.abs(all_results_u['values'][list(all_results_u['iters']).index(kernelshap_iters)][:,y])
        width_phi_u3 = arr_phi_u3*2
        isv_5 = ref_phi_U(all_results['values'][list(all_results['iters']).index(kernelshap_iters)], x, y, width_phi_u3, diff_u)


        # print("FENG MC")
        start=time.time()
        mc_m, mc_u = MonteCarlo_UNI(ind, X_train_s_TMP, modelRF, om_VV, surrogate_VV, M)
        time_F_MC.append(time.time()-start)
        time_H_MC.append(time.time()-start)
        # mc_m = MonteCarlo(ind, X_train_s_TMP, modelRF, om_VV, surrogate_VV) #OK
        isv_6 = ref_phi(mc_m, x, y, num_features, diff_u)
        
        # print("HAN MC")
        # mc_u = MonteCarlo_U(ind, X_train_s_TMP, modelRF, om_VV, surrogate_VV)
        arr_phi_u4=np.abs(mc_u[:,y])
        width_phi_u4 = arr_phi_u4*2
        isv_7 = ref_phi_U(mc_m, x, y, width_phi_u4, diff_u)
        


        # ISV_0: FENG FS  
        # ISV_1: FENG UKS
        # ISV_2: FENG KS
        # ISV_3: HAN FS
        # ISV_4: HAN UKS
        # ISV_5: HAN KS
        # ISV_6: FENG MC
        # ISV_7: HAN MC

        m0=np.mean(isv_0, axis=1)
        e0=(m0-isv_0[:,0])
        m1=np.mean(isv_1, axis=1)
        e1=(m1-isv_1[:,0])
        m2=np.mean(isv_2, axis=1)
        e2=(m2-isv_2[:,0])
        m3=np.mean(isv_3, axis=1)
        e3=(m3-isv_3[:,0])
        m4=np.mean(isv_4, axis=1)
        e4=(m4-isv_4[:,0])
        m5=np.mean(isv_5, axis=1)
        e5=(m5-isv_5[:,0])
        m6=np.mean(isv_6, axis=1)
        e6=(m6-isv_6[:,0])
        m7=np.mean(isv_7, axis=1)
        e7=(m7-isv_7[:,0])
        
        
        Eucl_F_UKS_FS.append(euclidean_distance(isv_1, isv_0))
        Eucl_F_UKS_MC.append(euclidean_distance(isv_1, isv_6))
        Eucl_F_UKS_KS.append(euclidean_distance(isv_1, isv_2))
        Eucl_H_UKS_FS.append(euclidean_distance(isv_4, isv_3))
        Eucl_H_UKS_MC.append(euclidean_distance(isv_4, isv_7))
        Eucl_H_UKS_KS.append(euclidean_distance(isv_4, isv_5))
        Eucl_F_H.append(euclidean_distance(isv_0, isv_3))
        
        average_F_UKS.append(m1)
        average_F_MC.append(m6)
        average_F_KS.append(m2)
        average_F_FS.append(m0)
        average_H_UKS.append(m4)
        average_H_MC.append(m7)
        average_H_KS.append(m5)
        average_H_FS.append(m3)
        
        error_F_UKS.append(e1)
        error_F_MC.append(e6)
        error_F_KS.append(e2)
        error_F_FS.append(e0)
        error_H_UKS.append(e4)
        error_H_MC.append(e7)
        error_H_KS.append(e5)
        error_H_FS.append(e3)

        
        L1_F_FS.append(np.linalg.norm(np.array(m1) - np.array(m0), ord=1))
        L1_F_KS.append(np.linalg.norm(np.array(m1) - np.array(m2), ord=1))
        L1_F_MC.append(np.linalg.norm(np.array(m1) - np.array(m6), ord=1))
        L1_H_FS.append(np.linalg.norm(np.array(m4) - np.array(m3), ord=1))
        L1_H_KS.append(np.linalg.norm(np.array(m4) - np.array(m5), ord=1))
        L1_H_MC.append(np.linalg.norm(np.array(m4) - np.array(m7), ord=1))

        L2_F_FS.append(np.linalg.norm(np.array(m1) - np.array(m0)))
        L2_F_KS.append(np.linalg.norm(np.array(m1) - np.array(m2)))
        L2_F_MC.append(np.linalg.norm(np.array(m1) - np.array(m6)))
        L2_H_FS.append(np.linalg.norm(np.array(m4) - np.array(m3)))
        L2_H_KS.append(np.linalg.norm(np.array(m4) - np.array(m5)))
        L2_H_MC.append(np.linalg.norm(np.array(m4) - np.array(m7)))

        L1_F_FS_err.append(np.linalg.norm(np.array(e1) - np.array(e0), ord=1))
        L1_F_KS_err.append(np.linalg.norm(np.array(e1) - np.array(e2), ord=1))
        L1_F_MC_err.append(np.linalg.norm(np.array(e1) - np.array(e6), ord=1))
        L1_H_FS_err.append(np.linalg.norm(np.array(e4) - np.array(e3), ord=1))
        L1_H_KS_err.append(np.linalg.norm(np.array(e4) - np.array(e5), ord=1))
        L1_H_MC_err.append(np.linalg.norm(np.array(e4) - np.array(e7), ord=1))

        L2_F_FS_err.append(np.linalg.norm(np.array(e1) - np.array(e0)))
        L2_F_KS_err.append(np.linalg.norm(np.array(e1) - np.array(e2)))
        L2_F_MC_err.append(np.linalg.norm(np.array(e1) - np.array(e6)))
        L2_H_FS_err.append(np.linalg.norm(np.array(e4) - np.array(e3)))
        L2_H_KS_err.append(np.linalg.norm(np.array(e4) - np.array(e5)))
        L2_H_MC_err.append(np.linalg.norm(np.array(e4) - np.array(e7)))
    except:
        pass
    
    # break
    
    #print all the computed metrics
    # Print the error rates

print("\tEuclidean Distances:")
print("\tF_UKS_FS:", np.mean(Eucl_F_UKS_FS))
print("\tF_UKS_MC:", np.mean(Eucl_F_UKS_MC))
print("\tF_UKS_KS:", np.mean(Eucl_F_UKS_KS))
print("\tH_UKS_FS:", np.mean(Eucl_H_UKS_FS))
print("\tH_UKS_MC:", np.mean(Eucl_H_UKS_MC))
print("\tH_UKS_KS:", np.mean(Eucl_H_UKS_KS))
print("\tF_H:", np.mean(Eucl_F_H))

print("")
print("\t","-"*100)
print("")

# Print the L2 distances
print("\tL2 distances:")
print("\tL2_F_FS:", np.mean(L2_F_FS))
print("\tL2_F_KS:", np.mean(L2_F_KS))
print("\tL2_F_MC:", np.mean(L2_F_MC))
print("\tL2_H_FS:", np.mean(L2_H_FS))
print("\tL2_H_KS:", np.mean(L2_H_KS))
print("\tL2_H_MC:", np.mean(L2_H_MC))

print("")

# Print the L2 distances on err
print("\tL2 distances on err:")
print("\tL2_F_FS_err:", np.mean(L2_F_FS_err))
print("\tL2_F_KS_err:", np.mean(L2_F_KS_err))
print("\tL2_F_MC_err:", np.mean(L2_F_MC_err))
print("\tL2_H_FS_err:", np.mean(L2_H_FS_err))
print("\tL2_H_KS_err:", np.mean(L2_H_KS_err))
print("\tL2_H_MC_err:", np.mean(L2_H_MC_err))

print("")
print("\t","-"*100)
print("")

# Print the L1 distances
print("\tL1 distances:")
print("\tL1_F_FS:", np.mean(L1_F_FS))
print("\tL1_F_KS:", np.mean(L1_F_KS))
print("\tL1_F_MC:", np.mean(L1_F_MC))
print("\tL1_H_FS:", np.mean(L1_H_FS))
print("\tL1_H_KS:", np.mean(L1_H_KS))
print("\tL1_H_MC:", np.mean(L1_H_MC))

print("")

# Print the L1 distances on err
print("\tL1 distances on err:")
print("\tL1_F_FS_err:", np.mean(L1_F_FS_err))
print("\tL1_F_KS_err:", np.mean(L1_F_KS_err))
print("\tL1_F_MC_err:", np.mean(L1_F_MC_err))
print("\tL1_H_FS_err:", np.mean(L1_H_FS_err))
print("\tL1_H_KS_err:", np.mean(L1_H_KS_err))
print("\tL1_H_MC_err:", np.mean(L1_H_MC_err))

print("")
print("\tInference Time:")
print("\tF_UKS:", np.mean(time_F_UKS))
print("\tF_FS:", np.mean(time_F_FS))
print("\tF_MC:", np.mean(time_F_MC))
print("\tF_KS:", np.mean(time_F_KS))
print("\tH_UKS:", np.mean(time_H_UKS))
print("\tH_FS:", np.mean(time_H_FS))
print("\tH_MC:", np.mean(time_H_MC))
print("\tH_KS:", np.mean(time_H_KS))


with open(f"results_LIKE/{dataset}_results_SEED={SEED}.txt", "a+") as file:
    file.write("#"*100)
    file.write("\tEuclidean Distances:\n")
    file.write(f"\tF_UKS_FS: {np.mean(Eucl_F_UKS_FS)}\n")
    file.write(f"\tF_UKS_MC: {np.mean(Eucl_F_UKS_MC)}\n")
    file.write(f"\tF_UKS_KS: {np.mean(Eucl_F_UKS_KS)}\n")
    file.write(f"\tH_UKS_FS: {np.mean(Eucl_H_UKS_FS)}\n")
    file.write(f"\tH_UKS_MC: {np.mean(Eucl_H_UKS_MC)}\n")
    file.write(f"\tH_UKS_KS: {np.mean(Eucl_H_UKS_KS)}\n")
    file.write(f"\tF_H: {np.mean(Eucl_F_H)}\n")
    # file.write("\n")
    file.write("-"*100)
    # file.write("\n")
    file.write("\tL2 distances:\n")
    file.write(f"\tL2_F_FS: {np.mean(L2_F_FS)}\n")
    file.write(f"\tL2_F_KS: {np.mean(L2_F_KS)}\n")
    file.write(f"\tL2_F_MC: {np.mean(L2_F_MC)}\n")
    file.write(f"\tL2_H_FS: {np.mean(L2_H_FS)}\n")
    file.write(f"\tL2_H_KS: {np.mean(L2_H_KS)}\n")
    file.write(f"\tL2_H_MC: {np.mean(L2_H_MC)}\n")
    # file.write("\n")
    file.write("\tL2 distances on err:\n")
    file.write(f"\tL2_F_FS_err: {np.mean(L2_F_FS_err)}\n")
    file.write(f"\tL2_F_KS_err: {np.mean(L2_F_KS_err)}\n")
    file.write(f"\tL2_F_MC_err: {np.mean(L2_F_MC_err)}\n")
    file.write(f"\tL2_H_FS_err: {np.mean(L2_H_FS_err)}\n")
    file.write(f"\tL2_H_KS_err: {np.mean(L2_H_KS_err)}\n")
    file.write(f"\tL2_H_MC_err: {np.mean(L2_H_MC_err)}\n")
    # file.write("\n")
    file.write("-"*100)
    # file.write("\n")
    file.write("\tL1 distances:\n")
    file.write(f"\tL1_F_FS: {np.mean(L1_F_FS)}\n")
    file.write(f"\tL1_F_KS: {np.mean(L1_F_KS)}\n")
    file.write(f"\tL1_F_MC: {np.mean(L1_F_MC)}\n")
    file.write(f"\tL1_H_FS: {np.mean(L1_H_FS)}\n")
    file.write(f"\tL1_H_KS: {np.mean(L1_H_KS)}\n")
    file.write(f"\tL1_H_MC: {np.mean(L1_H_MC)}\n")
    # file.write("\n")
    file.write("\tL1 distances on err:\n")
    file.write(f"\tL1_F_FS_err: {np.mean(L1_F_FS_err)}\n")
    file.write(f"\tL1_F_KS_err: {np.mean(L1_F_KS_err)}\n")
    file.write(f"\tL1_F_MC_err: {np.mean(L1_F_MC_err)}\n")
    file.write(f"\tL1_H_FS_err: {np.mean(L1_H_FS_err)}\n")
    file.write(f"\tL1_H_KS_err: {np.mean(L1_H_KS_err)}\n")
    file.write(f"\tL1_H_MC_err: {np.mean(L1_H_MC_err)}\n")


SAVE=True
if SAVE:
    # save all the metrics in a pickle file inside the folder 'dump'
    with open(f"dump_LIKE/{dataset}_results_SEED={SEED}_KI=128_TH=0.1.pkl", "wb") as file:
        pickle.dump({
            "Eucl_F_UKS_FS": Eucl_F_UKS_FS,
            "Eucl_F_UKS_MC": Eucl_F_UKS_MC,
            "Eucl_F_UKS_KS": Eucl_F_UKS_KS,
            "Eucl_H_UKS_FS": Eucl_H_UKS_FS,
            "Eucl_H_UKS_MC": Eucl_H_UKS_MC,
            "Eucl_H_UKS_KS": Eucl_H_UKS_KS,
            "Eucl_F_H": Eucl_F_H,
            "L2_F_FS": L2_F_FS,
            "L2_F_KS": L2_F_KS,
            "L2_F_MC": L2_F_MC,
            "L2_H_FS": L2_H_FS,
            "L2_H_KS": L2_H_KS,
            "L2_H_MC": L2_H_MC,
            "L2_F_FS_err": L2_F_FS_err,
            "L2_F_KS_err": L2_F_KS_err,
            "L2_F_MC_err": L2_F_MC_err,
            "L2_H_FS_err": L2_H_FS_err,
            "L2_H_KS_err": L2_H_KS_err,
            "L2_H_MC_err": L2_H_MC_err,
            "L1_F_FS": L1_F_FS,
            "L1_F_KS": L1_F_KS,
            "L1_F_MC": L1_F_MC,
            "L1_H_FS": L1_H_FS,
            "L1_H_KS": L1_H_KS,
            "L1_H_MC": L1_H_MC,
            "L1_F_FS_err": L1_F_FS_err,
            "L1_F_KS_err": L1_F_KS_err,
            "L1_F_MC_err": L1_F_MC_err,
            "L1_H_FS_err": L1_H_FS_err,
            "L1_H_KS_err": L1_H_KS_err,
            "L1_H_MC_err": L1_H_MC_err,
            "average_F_UKS": average_F_UKS,
            "average_F_MC": average_F_MC,
            "average_F_KS": average_F_KS,
            "average_F_FS": average_F_FS,
            "average_H_UKS": average_H_UKS,
            "average_H_MC": average_H_MC,
            "average_H_KS": average_H_KS,
            "average_H_FS": average_H_FS,
            "error_F_UKS": error_F_UKS,
            "error_F_MC": error_F_MC,
            "error_F_KS": error_F_KS,
            "error_F_FS": error_F_FS,
            "error_H_UKS": error_H_UKS,
            "error_H_MC": error_H_MC,
            "error_H_KS": error_H_KS,
            "error_H_FS": error_H_FS,
            "time_F_UKS": time_F_UKS,
            "time_F_MC": time_F_MC,
            "time_F_KS": time_F_KS,
            "time_F_FS": time_F_FS,
            "time_H_UKS": time_H_UKS,
            "time_H_MC": time_H_MC,
            "time_H_KS": time_H_KS,
            "time_H_FS": time_H_FS
        }, file)

    
