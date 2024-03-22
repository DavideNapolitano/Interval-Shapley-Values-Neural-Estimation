import torch
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data import RandomSampler, BatchSampler
from copy import deepcopy
from tqdm.auto import tqdm
import torch.nn as nn
import itertools
import numpy as np
import sys

sys.path.append('IntervalSV/code')

from utils import *

def validate_surr_VV(surrogate, loss_1, loss_2, data_loader, alpha, beta, batch_size):

    with torch.no_grad():
        # Setup.
        device = next(surrogate.surrogate.parameters()).device
        mean_loss = 0
        mean_loss1=0
        mean_loss2=0
        mean_ne=0
        N = 0
        #counter_errors_n=0
        for x, v1, v2, S in data_loader:
            counter_errors_n=0
            counter_errors_p=0
            x = x.to(device)
            v1 = v1.to(device)
            v2 = v2.to(device)
            S = S.to(device)
            pred1, pred2 = surrogate(x, S)
            pred1_soft=pred1.softmax(dim=-1)
            pred2_soft=pred2.softmax(dim=-1)
            p1_n=pred1_soft.detach().numpy()
            p2_n=pred2_soft.detach().numpy()
            for el1, el2 in zip(p1_n, p2_n):
                if el1[0]>el2[1]:
                    counter_errors_n+=1
                if el2[0]>el1[1]:
                    counter_errors_p+=1
            loss1_Val = loss_1(pred1, v1)
            loss2_Val = loss_2(pred2, v2)
            #loss=alpha*loss1+beta*loss2
            ce=np.mean([counter_errors_n,counter_errors_p])
            loss = loss1_Val + loss2_Val + ce/len(x)
            mean_ne+=ce
            N += len(x)
            mean_loss += len(x) * (loss - mean_loss) / N
            mean_loss1 += len(x) * (loss1_Val - mean_loss1) / N
            mean_loss2 += len(x) * (loss2_Val - mean_loss2) / N

    return mean_loss, mean_loss1, mean_loss2, mean_ne/N


def generate_labels_VV(dataset, inner_model, batch_size):

    with torch.no_grad():
        # Setup.
        preds_v1 = []
        preds_v2=[]
        if isinstance(inner_model, torch.nn.Module):
            device = next(inner_model.parameters()).device
        else:
            device = torch.device('cpu')
        loader = DataLoader(dataset, batch_size=batch_size)

        for (x,) in loader:
            pred_v1, pred_v2 = inner_model(x.to(device))#.cpu()
            preds_v1.append(pred_v1)
            preds_v2.append(pred_v2)

    return torch.cat(preds_v1), torch.cat(preds_v2)


class Surrogate_VV:

    def __init__(self, surrogate, num_features, groups=None):
        # Store surrogate model.
        self.surrogate = surrogate
        self.used_loss1= None
        self.used_loss1= None

        # Store feature groups.
        if groups is None:
            self.num_players = num_features
            self.groups_matrix = None
        else:
            # Verify groups.
            inds_list = []
            for group in groups:
                inds_list += list(group)
            assert np.all(np.sort(inds_list) == np.arange(num_features))

            # Map groups to features.
            self.num_players = len(groups)
            device = next(surrogate.parameters()).device
            self.groups_matrix = torch.zeros(len(groups), num_features, dtype=torch.float32, device=device)
            for i, group in enumerate(groups):
                self.groups_matrix[i, group] = 1

        
    def train_original_model_VV(self,
                             train_data,
                             val_data,
                             original_model_VV,
                             batch_size,
                             max_epochs,
                             loss_fn1,
                             loss_fn2,
                             alpha=1,
                             beta=1,
                             validation_samples=1,
                             validation_batch_size=None,
                             lr=1e-3,
                             min_lr=1e-5,
                             lr_factor=0.5,
                             lookback=5,
                             weight_decay=0,
                             training_seed=None,
                             validation_seed=None,
                             bar=False,
                             verbose=False,
                             debug=False):

        # Set up train dataset.
        if isinstance(train_data, np.ndarray):
            train_data = torch.tensor(train_data, dtype=torch.float32)
        if isinstance(train_data, torch.Tensor):
            train_set = TensorDataset(train_data)
        elif isinstance(train_data, Dataset):
            train_set = train_data
        else:
            raise ValueError('train_data must be either tensor or a PyTorch Dataset')
        
        self.used_loss1=loss_fn1
        self.used_loss2=loss_fn2
          
        print("LEN TRAIN DATA:",len(train_data),"LEN TRAIN SET:",len(train_set))

        # Set up train data loader.
        # RANDOM SAMPLER DA TORCH
        print("NUM_SAMPLES:",int(np.ceil(len(train_set) / batch_size))*batch_size)

        random_sampler = RandomSampler( train_set, replacement=True, num_samples=int(np.ceil(len(train_set) / batch_size))*batch_size)
        print("Random Sampler", random_sampler)
        batch_sampler = BatchSampler(random_sampler, batch_size=batch_size, drop_last=True)
        print("Batch Sampler", batch_sampler)
        train_loader = DataLoader(train_set, batch_sampler=batch_sampler)
        print("Train Loader:",len(train_loader),len(train_loader)*batch_size)

        # Set up validation dataset.
        sampler = UniformSampler(self.num_players) #USATO PER CREARE LA MASCHERA

        if validation_seed is not None:
            torch.manual_seed(validation_seed)
        S_val = sampler.sample(len(val_data) * validation_samples)
        if validation_batch_size is None:
            validation_batch_size = batch_size

        if isinstance(val_data, np.ndarray):
            val_data = torch.tensor(val_data, dtype=torch.float32)

        if isinstance(val_data, torch.Tensor):
            print("IS INSTANCE")
            # Generate validation labels.
            y_val_v1, y_val_v2 = generate_labels_VV(TensorDataset(val_data), original_model_VV, validation_batch_size) ######################
            y_val_v1_repeat = y_val_v1.repeat(validation_samples, *[1 for _ in y_val_v1.shape[1:]])
            y_val_v2_repeat = y_val_v2.repeat(validation_samples, *[1 for _ in y_val_v2.shape[1:]])

            # Create dataset.
            val_data_repeat = val_data.repeat(validation_samples, 1)
            val_set = TensorDataset(val_data_repeat, y_val_v1_repeat, y_val_v2_repeat, S_val) ###############################################
        elif isinstance(val_data, Dataset):
            print("NOT INSTANCE")
            # Generate validation labels.
            y_val = generate_labels(val_data, original_model, validation_batch_size)
            y_val_repeat = y_val.repeat(validation_samples, *[1 for _ in y_val.shape[1:]])

            # Create dataset.
            val_set = DatasetRepeat([val_data, TensorDataset(y_val_repeat, S_val)])
        else:
            raise ValueError('val_data must be either tuple of tensors or a PyTorch Dataset')

        val_loader = DataLoader(val_set, batch_size=validation_batch_size)

        # Setup for training.
        surrogate = self.surrogate
        device = next(surrogate.parameters()).device
        #optimizer = optim.Adam(surrogate.parameters(), lr=lr)
        optimizer = optim.AdamW(surrogate.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau( optimizer, factor=lr_factor, patience=lookback//2, min_lr=min_lr, verbose=verbose)
        #print("LOSS WEIGH - Alpha:",alpha,"Beta:",beta)
        best_loss = np.inf
        best_epoch = 0
        best_model = deepcopy(surrogate)
        loss_list = [best_loss]

        if training_seed is not None:
            torch.manual_seed(training_seed)

        print("STARTING TRAINING PHASE")
        val_loss_prob=[]
        val_loss_prob_L=[]
        val_loss_prob_U=[]
        train_loss_prob=[]
        train_loss_prob_L=[]
        train_loss_prob_U=[]
        ep_idx=0

        for epoch in range(max_epochs):
            mean_loss_T = 0
            mean_loss1_T=0
            mean_loss2_T=0
            mean_ne_T=0
            Nt = 0
            
            # Batch iterable.
            if bar:
                batch_iter = tqdm(train_loader, desc='Training epoch')
            else:
                batch_iter = train_loader
            itr_idx=0
            for (x,) in batch_iter:
                ne_n=0
                ne_p=0
                # Prepare data.
                x = x.to(device)
                if debug and itr_idx==0 and ep_idx==0:
                    print("DATA:",x.shape)

                # Get original model prediction.
                with torch.no_grad():
                    v1, v2 = original_model_VV(x)
                    if debug and itr_idx==0 and ep_idx==0:
                        print("V1:",v1.shape,"V2:", v2.shape)

                
                # Generate subsets.
                S = sampler.sample(batch_size).to(device=device)
                if debug and itr_idx==0 and ep_idx==0:
                    print("SUBSET:",S.shape)

                # Make predictions.
                pred1, pred2 = self.__call__(x, S)
                if debug and itr_idx==0 and ep_idx==0:
                    print("PRED_V1:",pred_v1.shape,"PRED_V2:", pred_v2.shape)
                    print("PRED_L:",pred_v1)
                    print("V1:",v1)
                    print("PRED_U:",pred_v2)
                    print("V2:",v2)
                    #print(type(pred_L), type(pred_U), type(y_L), type(y_U))
                    #print(type(pred_L[0]), type(pred_U[0]), type(y_L[0]), type(y_U[0]))
                    #print(type(pred_L[0][0]), type(pred_U[0][0]), type(y_L[0][0]), type(y_U[0][0]))
                
                pred1_soft=pred1.softmax(dim=-1)
                pred2_soft=pred2.softmax(dim=-1)
                p1_n=pred1_soft.detach().numpy()
                p2_n=pred2_soft.detach().numpy()
                for el1, el2 in zip(p1_n, p2_n):
                    if el1[0]>el2[1]:
                        ne_n+=1
                    if el2[0]>el1[1]:
                        ne_p+=1
                ne=np.mean([ne_n, ne_p])
                
                loss1_train = loss_fn1(pred1, v1)
                loss2_train= loss_fn2(pred2, v2)

                loss = alpha*loss1_train + beta*loss2_train + ne/batch_size
                #loss = (1+ne/bath_size)*(alpha*loss1 + beta*loss2)

                Nt += len(x)
                mean_loss_T += len(x) * (loss - mean_loss_T) / Nt
                mean_loss1_T += len(x) * (loss1_train - mean_loss1_T) / Nt
                mean_loss2_T += len(x) * (loss2_train - mean_loss2_T) / Nt
                mean_ne_T+=ne


                # Optimizer step.
                loss.backward()
                optimizer.step()
                surrogate.zero_grad()
                itr_idx+=1
            mean_ne_T=mean_ne_T/Nt
            
            train_loss_prob.append(mean_loss_T.item())
            train_loss_prob_L.append(mean_loss1_T.item())
            train_loss_prob_U.append(mean_loss2_T.item())
        
            # Evaluate validation loss.
            self.surrogate.eval()
            val_loss, val_loss1, val_loss2, mean_ne_V= validate_surr_VV(self, loss_fn1,loss_fn2, val_loader, alpha, beta, batch_size)#.item() ############################################################## MODIFY
            self.surrogate.train()
            
            val_loss_prob.append(val_loss)
            val_loss_prob_L.append(val_loss1)
            val_loss_prob_U.append(val_loss2)
            
            # Print progress.
            if verbose:
                print('----- Epoch = {} -----'.format(epoch + 1))
                print('Train loss = {:.8f}'.format(mean_loss_T))
                print('Train loss L = {:.8f}'.format(mean_loss1_T))
                print('Train loss U = {:.8f}'.format(mean_loss2_T))
                print("Train mean NE: {:.8f}".format(mean_ne_T))
                print('Val loss = {:.8f}'.format(val_loss))
                print('Val loss L = {:.8f}'.format(val_loss1))
                print('Val loss U = {:.8f}'.format(val_loss2))
                print("Val mean NE: {:.8f}".format(mean_ne_V))
                print('')
                
            scheduler.step(val_loss)
            loss_list.append(val_loss)

            # Check if best model.
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(surrogate)
                best_epoch = epoch
                if verbose:
                    print('New best epoch, loss = {:.8f}'.format(val_loss))
                    print('')
            elif epoch - best_epoch == lookback:
                if verbose:
                    print('Stopping early')
                break
            ep_idx+=1
            
        # Clean up.
        for param, best_param in zip(surrogate.parameters(), best_model.parameters()):
            param.data = best_param.data
        self.loss_list = loss_list
        self.surrogate.eval()
        
        
    def __call__(self, x, S):
        if self.groups_matrix is not None:
            print("GROUP MATRIX")
            S = torch.mm(S, self.groups_matrix)

        return self.surrogate((x, S))