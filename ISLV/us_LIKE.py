import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
from copy import deepcopy
from tqdm.auto import tqdm

import sys

sys.path.append('IntervalSV/code')
from utils import *


# MOORE SUBTRACTION --> SAME SINCE ONLY A VALUE
def additive_efficient_normalization_U(pred, grand, null):
    gap = (grand + null) - torch.sum(pred, dim=1)
    # gap = gap.detach()
    return torch.abs(pred + gap.unsqueeze(1) / pred.shape[1])

# OK
def evaluate_explainer_U(explainer, normalization, x, grand, null, num_players, inference=False):

    # Evaluate explainer.
    pred = explainer(x)

    # Reshape SHAP values.
    if len(pred.shape) == 4:
        # Image.
        image_shape = pred.shape
        pred = pred.reshape(len(x), -1, num_players)
        pred = pred.permute(0, 2, 1)
    else:
        # Tabular.
        image_shape = None
        pred = pred.reshape(len(x), num_players, -1)

    # For pre-normalization efficiency gap.
    total = pred.sum(dim=1)

    # Apply normalization.
    if normalization:
        pred = normalization(pred, grand, null)

    # Reshape for inference.
    if inference:
        if image_shape is not None:
            pred = pred.permute(0, 2, 1)
            pred = pred.reshape(image_shape)

        return pred

    return pred, total

# OK
def calculate_grand_coalition_U(dataset, imputer, batch_size, link, device, num_workers):
    # print("COMP GRAND COAL")
    ones = torch.ones(batch_size, imputer.num_players, dtype=torch.float32, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    with torch.no_grad():
        grand = []
        for (x,) in loader:
            v1, v2 = imputer(x.to(device), ones[:len(x)])
            v1 = link(v1)
            v2 = link(v2)
            v1=v1.data.numpy()
            v2=v2.data.numpy()
            # print("V1, V2",v1.shape, v2.shape)
            # print(v1, v2)
            v_neg=np.array([v1[:,0],v2[:,1]]).T
            v_pos=np.array([v2[:,0],v1[:,1]]).T
            # print("N P", v_neg.shape, v_pos.shape)
            # print(v_neg, v_pos)
            pred=[]
            for el1, el2 in zip(v_neg, v_pos):
                pred.append([np.abs(el1[1]-el1[0])/2, np.abs(el2[1]-el2[0])/2])
            pred= torch.tensor(pred, dtype=torch.float)
            # print("PRED", pred.shape)
            # print(pred)
            # print("")
            grand.append(pred) 

        # Concatenate and return.
        grand = torch.cat(grand)
        if len(grand.shape) == 1:
            grand = grand.unsqueeze(1)

    return grand

# OK
def generate_validation_data_U(val_set, imputer, validation_samples, sampler, batch_size, link, device, num_workers):
    # Generate coalitions.
    val_S = sampler.sample( validation_samples * len(val_set), paired_sampling=True).reshape(  len(val_set), validation_samples, imputer.num_players)

    # Get values.
    val_values = []
    for i in range(validation_samples):
        # Set up data loader.
        dset = DatasetRepeat([val_set, TensorDataset(val_S[:, i])])
        loader = DataLoader(dset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
        
        values = []
        for x, S in loader:
            v1, v2 = imputer(x.to(device), S.to(device))
            v1 = link(v1)
            v2 = link(v2)
            v1=v1.data.numpy()
            v2=v2.data.numpy()
            v_neg=np.array([v1[:,0],v2[:,1]]).T
            v_pos=np.array([v2[:,0],v1[:,1]]).T
            pred=[]
            for el1, el2 in zip(v_neg, v_pos):
                pred.append([np.abs(el1[1]-el1[0])/2, np.abs(el2[1]-el2[0])/2])
            pred= torch.tensor(pred, dtype=torch.float)
            
            values.append(pred)

        val_values.append(torch.cat(values))

    val_values = torch.stack(val_values, dim=1)
    return val_S, val_values

# OK
def validate_U(val_loader, imputer, explainer, null, link, normalization):
    with torch.no_grad():
        # Setup.
        device = next(explainer.parameters()).device
        mean_loss = 0
        mean_loss_tot=0
        N = 0
        CE = 0
        loss_fn = nn.MSELoss()

        for x, grand, S, values in val_loader:
            # Move to device.
            x = x.to(device)
            S = S.to(device)
            grand = grand.to(device)
            values = values.to(device)

            # Evaluate explainer.
            pred, _ = evaluate_explainer_U(explainer, normalization, x, grand, null, imputer.num_players)

            pred_n=pred.cpu().data.numpy()

            CE_N=[]
            CE_P=[]
            for el in pred_n:
                c_n_p=0
                c_n_n=0
                for out_feat in el:
                    if out_feat[0]<0:
                        c_n_n+=1
                    if out_feat[1]<0:
                        c_n_p+=1
                CE_N.append(c_n_n)
                CE_P.append(c_n_p)
            CE_N_AVG=np.mean(CE_N)
            CE_P_AVG=np.mean(CE_P)
            CE_AVG=np.mean([CE_N_AVG, CE_P_AVG])

            # Calculate loss.
            approx = null + torch.matmul(S, pred)
            loss = loss_fn(approx, values)
            loss_tot = loss + CE_AVG/len(x)

            # Update average.
            N += len(x)
            CE += len(x) * (CE_AVG/len(x) - CE) / N
            # CE += CE_AVG/len(x)
            mean_loss += len(x) * (loss - mean_loss) / N
            mean_loss_tot += len(x) * (loss_tot - mean_loss_tot) / N

    return mean_loss_tot, mean_loss, CE


class FastSHAP_U:
    def __init__(self,
                 explainer,
                 imputer,
                 normalization='none',
                 link=None):
        # Set up explainer, imputer and link function.
        self.explainer = explainer
        self.imputer = imputer
        self.num_players = imputer.num_players
        self.null = None
        if link is None or link == 'none':
            self.link = nn.Identity()
        elif isinstance(link, nn.Module):
            self.link = link
        else:
            raise ValueError('unsupported link function: {}'.format(link))

        # Set up normalization.
        if normalization is None or normalization == 'none':
            self.normalization = None
        elif normalization == 'additive':
            self.normalization = additive_efficient_normalization_U
        # elif normalization == 'multiplicative':
        #     self.normalization = multiplicative_efficient_normalization
        else:
            raise ValueError('unsupported normalization: {}'.format(normalization))

    def train(self,
              train_data,
              val_data,
              batch_size,
              num_samples,
              max_epochs,
              lr=2e-4,
              weight_decay=0.01,
              min_lr=1e-5,
              lr_factor=0.5,
              eff_lambda=0,
              paired_sampling=True,
              validation_samples=None,
              lookback=5,
              training_seed=None,
              validation_seed=None,
              num_workers=0,
              bar=False,
              verbose=False):

        # Set up explainer model.
        explainer = self.explainer
        num_players = self.num_players
        imputer = self.imputer
        link = self.link
        normalization = self.normalization
        explainer.train()
        device = next(explainer.parameters()).device

        # Verify other arguments.
        if validation_samples is None:
            validation_samples = num_samples

        # Set up train dataset.
        if isinstance(train_data, np.ndarray):
            x_train = torch.tensor(train_data, dtype=torch.float32)
            train_set = TensorDataset(x_train)
        elif isinstance(train_data, torch.Tensor):
            train_set = TensorDataset(train_data)
        elif isinstance(train_data, Dataset):
            train_set = train_data
        else:
            raise ValueError('train_data must be np.ndarray, torch.Tensor or Dataset')

        # Set up validation dataset.
        if isinstance(val_data, np.ndarray):
            x_val = torch.tensor(val_data, dtype=torch.float32)
            val_set = TensorDataset(x_val)
        elif isinstance(val_data, torch.Tensor):
            val_set = TensorDataset(val_data)
        elif isinstance(val_data, Dataset):
            val_set = val_data
        else:
            raise ValueError('train_data must be np.ndarray, torch.Tensor or Dataset')

        # Grand coalition value.
        grand_train = calculate_grand_coalition_U(train_set, imputer, batch_size * num_samples, link, device, num_workers).cpu()
        grand_val = calculate_grand_coalition_U(val_set, imputer, batch_size * num_samples, link, device, num_workers).cpu()

        # Null coalition.
        with torch.no_grad():
            zeros = torch.zeros(1, num_players, dtype=torch.float32, device=device)
            
            v1, v2 = imputer(train_set[0][0].unsqueeze(0).to(device), zeros)
            v1 = link(v1[0])
            v2 = link(v2[0])
            v1=v1.data.numpy()
            v2=v2.data.numpy()
            v_neg=np.array([v1[0],v2[1]])
            v_pos=np.array([v2[0],v1[1]])
            null=[np.abs(v_neg[1]-v_neg[0])/2, np.abs(v_pos[1]-v_pos[0])/2]
            null=torch.tensor(null, dtype=torch.float)
            
            # null = link(imputer(train_set[0][0].unsqueeze(0).to(device), zeros)) # TODO
            if len(null.shape) == 1:
                # null = null.reshape(1, 1)
                null = null.unsqueeze(0)
        self.null = null
        # print("NULL", null.shape, null)

        # Set up train loader.
        train_set = DatasetRepeat([train_set, TensorDataset(grand_train)])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=num_workers)

        # Generate validation data.
        sampler = ShapleySampler(num_players)
        if validation_seed is not None:
            torch.manual_seed(validation_seed)

        val_S, val_values = generate_validation_data_U(val_set, imputer, validation_samples, sampler, batch_size * num_samples, link, device, num_workers)

        # Set up val loader.
        val_set = DatasetRepeat([val_set, TensorDataset(grand_val, val_S, val_values)])
        val_loader = DataLoader(val_set, batch_size=batch_size * num_samples, pin_memory=True, num_workers=num_workers)

        # Setup for training.
        loss_fn = nn.MSELoss()
        optimizer = optim.AdamW(explainer.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_factor, patience=lookback // 2, min_lr=min_lr, verbose=verbose)
        self.loss_list = []
        best_loss = np.inf
        best_epoch = -1
        best_model = None
        if training_seed is not None:
            torch.manual_seed(training_seed)


        for epoch in range(max_epochs):
            # Batch iterable.
            if bar:
                batch_iter = tqdm(train_loader, desc='Training epoch')
            else:
                batch_iter = train_loader

            mean_loss_T = 0
            mean_loss_tot_T=0
            N_T = 0
            CE_T = 0

            for x, grand in batch_iter:
                # Sample S.
                S = sampler.sample(batch_size * num_samples, paired_sampling=paired_sampling)

                # Move to device.
                x = x.to(device)
                S = S.to(device)
                grand = grand.to(device)

                # Evaluate value function.
                x_tiled = x.unsqueeze(1).repeat(1, num_samples, *[1 for _ in range(len(x.shape) - 1)]).reshape(batch_size * num_samples, *x.shape[1:])
                with torch.no_grad():
                    # values = link(imputer(x_tiled, S)) # OK
                    v1, v2 = imputer(x_tiled, S)
                    v1 = link(v1)
                    v2 = link(v2)
                    v1=v1.data.numpy()
                    v2=v2.data.numpy()
                    v_neg=np.array([v1[:,0],v2[:,1]]).T
                    v_pos=np.array([v2[:,0],v1[:,1]]).T
                    pred=[]
                    for el1, el2 in zip(v_neg, v_pos):
                        pred.append([np.abs(el1[1]-el1[0])/2, np.abs(el2[1]-el2[0])/2])
                    values= torch.tensor(pred, dtype=torch.float)
                    # print("VALUES:", values.shape)
                    # print("GRAND:", grand.shape, grand.dtype)

                # Evaluate explainer.
                pred, total = evaluate_explainer_U(explainer, normalization, x, grand, null, num_players)

                pred_n=pred.cpu().data.numpy()
                CE_N=[]
                CE_P=[]
                for el in pred_n:
                    c_n_p=0
                    c_n_n=0
                    for out_feat in el:
                        if out_feat[0]<0:
                            c_n_n+=1
                        if out_feat[1]<0:
                            c_n_p+=1
                    CE_N.append(c_n_n)
                    CE_P.append(c_n_p)
                CE_N_AVG=np.mean(CE_N)
                CE_P_AVG=np.mean(CE_P)
                CE_AVG=np.mean([CE_N_AVG, CE_P_AVG])

                # Calculate loss.
                S = S.reshape(batch_size, num_samples, num_players)
                values = values.reshape(batch_size, num_samples, -1)
                # print(null.dtype, S.dtype, pred.dtype)
                approx = null + torch.matmul(S, pred) # SUM
                # print(approx.dtype, values.dtype)
                loss = loss_fn(approx, values)

                # Take gradient step.
                loss_tot =  num_players * (loss + CE_AVG/len(x))
                loss_tot.backward()
                optimizer.step()
                explainer.zero_grad()

                N_T += len(x)
                CE_T += len(x) * (CE_AVG/len(x) - CE_T) / N_T
                mean_loss_T += len(x) * (loss - mean_loss_T) / N_T
                mean_loss_tot_T += len(x) * (loss_tot - mean_loss_tot_T) / N_T

            # Evaluate validation loss.
            explainer.eval()
            val_loss_tot, val_loss, CE_V =  validate_U(val_loader, imputer, explainer, null, link, normalization)#.item()
            val_loss_tot = val_loss_tot * num_players
            val_loss = val_loss * num_players
            CE_V = CE_V * num_players
            explainer.train()

            # Save loss, print progress.
            if verbose:
                print('----- Epoch = {} -----'.format(epoch + 1))
                print('Train loss tot = {:.6f}'.format(mean_loss_tot_T))
                print('Train loss = {:.6f}'.format(mean_loss_T*num_players))
                print('Train CE = {:.6f}'.format(CE_T*num_players))
                print('Val loss tot = {:.6f}'.format(val_loss_tot))
                print('Val loss = {:.6f}'.format(val_loss))
                print('Val CE = {:.6f}'.format(CE_V))
                print('')
            scheduler.step(val_loss_tot)
            self.loss_list.append(val_loss_tot)

            # Check for convergence.
            if self.loss_list[-1] < best_loss:
                best_loss = self.loss_list[-1]
                best_epoch = epoch
                best_model = deepcopy(explainer)
                if verbose:
                    print('New best epoch, loss = {:.6f}'.format(val_loss_tot))
                    print('')
            elif epoch - best_epoch == lookback:
                if verbose:
                    print('Stopping early at epoch = {}'.format(epoch))
                break

        # Copy best model.
        for param, best_param in zip(explainer.parameters(), best_model.parameters()):
            param.data = best_param.data
        explainer.eval()

    def shap_values(self, x):
        # Data conversion.
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        elif isinstance(x, torch.Tensor):
            pass
        else:
            raise ValueError('data must be np.ndarray or torch.Tensor')

        
        # Ensure null coalition is calculated.
        device = next(self.explainer.parameters()).device
        if self.null is None:
            with torch.no_grad():
                zeros = torch.zeros(1, self.num_players, dtype=torch.float32, device=device)
                
                v1, v2 = self.imputer(x[:1].to(device), zeros)
                v1 = self.link(v1[0])
                v2 = self.link(v2[0])
                v1=v1.data.numpy()
                v2=v2.data.numpy()
                v_neg=np.array([v1[0],v2[1]])
                v_pos=np.array([v2[0],v1[1]])
                null=[np.abs(v_neg[1]-v_neg[0])/2, np.abs(v_pos[1]-v_pos[0])/2]
                null=torch.tensor(null, dtype=torch.float)
                if len(null.shape) == 1:
                    # null = null.reshape(1, 1)
                    null = null.unsqueeze(0)
                
                # null = self.link(self.imputer(x[:1].to(device), zeros))
            # if len(null.shape) == 1:
            #     null = null.reshape(1, 1)
            self.null = null

        # Generate explanations.
        with torch.no_grad():
            # Calculate grand coalition (for normalization).
            if self.normalization:
                grand = calculate_grand_coalition_U(x, self.imputer, len(x), self.link, device, 0)
            else:
                grand = None

            # Evaluate explainer.
            x = x.to(device)
            pred = evaluate_explainer_U(self.explainer, self.normalization, x, grand, self.null, self.imputer.num_players, inference=True)
            
        pred=pred.cpu().data.numpy()

        return pred 