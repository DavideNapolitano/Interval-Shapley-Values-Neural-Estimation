from itertools import chain, combinations
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

link=nn.Softmax(dim=-1)
# create a function that given a binary mask, returns the corresponding subset of size-1
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def get_subsets(mask):
    how_many=np.sum(mask)
    all_subsets=list(powerset(np.where(mask==1)[0]))[:-1]
    rev=all_subsets[::-1]
    if int(how_many)!=1:
        new_subsets=rev[:int(how_many)]
    else:
        new_subsets=rev[:int(how_many)+1]
    return new_subsets

def adjust_to_sm(x, all_subsets, model_eval, num_features, target):
    #build dict intervals
    dict_intervals={}
    for el in all_subsets:
        tmp=np.zeros(num_features, dtype=np.int64)
        np.put(tmp, list(el), np.ones(len(el), dtype=np.int64))
        tmp_S=torch.tensor(tmp)
        tmp_S=tmp_S.unsqueeze(0)
        key="".join(str(el) for el in tmp)
        out=model_eval(x,tmp_S)
        v1=link(out[0]).data.numpy()[0]
        v2=link(out[1]).data.numpy()[0]
        if v1[0]<=v2[1]:
            v_neg=np.array([v1[0],v2[1]], dtype=np.float32)
        else:
            v_neg=np.array([v2[1],v1[0]], dtype=np.float32)
        if v2[0]<=v1[1]:
            v_pos=np.array([v2[0],v1[1]], dtype=np.float32)
        else:
            v_pos=np.array([v1[1],v2[0]], dtype=np.float32)
        dict_intervals[key]=[v_neg,v_pos]
        
    #adjust intervals
    adjusted_intervals={}
    for el in all_subsets:

        tmp=np.zeros(num_features, dtype=np.int64)
        np.put(tmp, list(el), np.ones(len(el), dtype=np.int64))
        key="".join(str(el) for el in tmp)

        if np.sum(tmp)!=0:
            values=dict_intervals[key]
            delta_neg=np.abs(values[0][1]-values[0][0])
            delta_pos=np.abs(values[1][1]-values[1][0])
            children = get_subsets(tmp)
            biggest_error=0
            for child in children:
                tmp_child=np.zeros(num_features, dtype=np.int64)
                np.put(tmp_child, list(child), np.ones(len(child), dtype=np.int64))
                child_key="".join(str(el) for el in tmp_child)
                values_child=adjusted_intervals[child_key] #Since bottom-up, child already adjusted
                delta_neg_child=np.abs(values_child[0][1]-values_child[0][0])
                delta_pos_child=np.abs(values_child[1][1]-values_child[1][0])
                if target==0: #target fixed based on grand coalition
                    if delta_neg_child>delta_neg:
                        if delta_neg_child-delta_neg>biggest_error:
                            biggest_error=np.abs(delta_neg_child-delta_neg)
                else:
                    if delta_pos_child>delta_pos:
                        if delta_pos_child-delta_pos>biggest_error:
                            biggest_error=np.abs(delta_pos_child-delta_pos)

            eps=biggest_error/2
            if biggest_error!=0:
                if target==0:
                    lower_neg=np.max([values[0][0]-eps, 0]) #
                    upper_neg=np.min([values[0][1]+eps, 1]) #
                    lower_pos=np.max([values[1][0]-eps, 0])
                    upper_pos=np.min([values[1][1]+eps, 1])
                    
                    old_delta=np.abs(values[0][1]-values[0][0])
                    new_delta=np.abs(upper_neg-lower_neg)
                    if new_delta - old_delta < biggest_error: # NEED SHIFT
                        shift = biggest_error - (new_delta - old_delta) 
                        if values[0][1]+eps > 1: # [ 1] [0 ]
                            lower_neg-=shift
                            upper_pos+=shift
                        if values[0][0]-eps < 0:# [0 ] [ 1]
                            lower_pos-=shift
                            upper_neg+=shift

                    adjusted_intervals[key]=[np.array([lower_neg, upper_neg]), np.array([lower_pos, upper_pos])]
                else:
                    lower_neg=np.max([values[0][0]-eps, 0])
                    upper_neg=np.min([values[0][1]+eps, 1])
                    lower_pos=np.max([values[1][0]-eps, 0]) #
                    upper_pos=np.min([values[1][1]+eps, 1]) #
                    
                    old_delta=np.abs(values[1][1]-values[1][0])
                    new_delta=np.abs(upper_pos-lower_pos)
                    if new_delta - old_delta < biggest_error:
                        shift = biggest_error - (new_delta - old_delta) 
                        if values[1][1]+eps > 1:# [0 ] [ 1]
                            lower_pos-=shift
                            upper_neg+=shift
                        if values[1][0]-eps < 0:# [ 1] [0 ]
                            lower_neg-=shift
                            upper_pos+=shift      
                    adjusted_intervals[key]=[np.array([lower_neg, upper_neg]), np.array([lower_pos, upper_pos])]
            else:
                adjusted_intervals[key]=dict_intervals[key]
        else:
            adjusted_intervals[key]=dict_intervals[key]
    
    return dict_intervals, adjusted_intervals        

class DictionaryData:
    def __init__(self, surrogate_VV_SM, num_features, all_subsets):
        self.surrogate_VV_SM=surrogate_VV_SM
        self.num_features=num_features
        self.all_subsets=all_subsets    

    def get_data_dict(self, data_loader):
        L1=[]
        L2=[]
        IDX=0
        COUNTER=0
        
        DICT_DATA={}
        for x in tqdm(data_loader):
            
            DATA_KEY="".join(str(val) for val in x.data.numpy()[0])
            S_one=torch.ones_like(x)
            out_1=self.surrogate_VV_SM(x,S_one) ##########################################################################
            v1=link(out_1[0]).data.numpy()[0]
            v2=link(out_1[1]).data.numpy()[0]
            v_neg=[v1[0],v2[1]]
            v_pos=[v2[0],v1[1]]    
            pred = [np.mean(v_neg), np.mean(v_pos)]
            target=np.argmax(pred)
            original_int, adjusted_int=adjust_to_sm(x, self.all_subsets, self.surrogate_VV_SM, self.num_features, target) #######################################
            DICT_DATA[DATA_KEY]=[original_int, adjusted_int]
            ERR1=[]
            ERR2=[]
            
            for el in self.all_subsets:
                tmp=np.zeros(self.num_features, dtype=np.int64)
                np.put(tmp, list(el), np.ones(len(el), dtype=np.int64))
                key="".join(str(val) for val in tmp)
                or_neg=original_int[key][0]
                or_pos=original_int[key][1]
                adj_neg=adjusted_int[key][0]
                adj_pos=adjusted_int[key][1]
                
                if target==0:
                    err2=np.linalg.norm(or_neg-adj_neg)
                    err1=np.linalg.norm(or_neg-adj_neg, ord=1)
                else:
                    err2=np.linalg.norm(or_pos-adj_pos)
                    err1=np.linalg.norm(or_pos-adj_pos, ord=1)
                COUNTER+=1
                ERR1.append(err1)
                ERR2.append(err2)
            L1.append(np.mean(ERR1))
            L2.append(np.mean(ERR2))
            IDX+=1

        print("L1", np.mean(L1))
        print("L2", np.mean(L2))   
        print("COUNTER", COUNTER)

        return DICT_DATA