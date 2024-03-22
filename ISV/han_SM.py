import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
from copy import deepcopy
from tqdm.auto import tqdm

import sys

sys.path.append('IntervalSV/code')
from utils import *


def rearrange_vectors(input_vector):
    vector_1=np.array([input_vector[0][0], input_vector[1][1]])
    vector_2=np.array([input_vector[1][0], input_vector[0][1]])
    return vector_1, vector_2

# MOORE SUBTRACTION --> SAME SINCE ONLY A VALUE
def additive_efficient_normalization_U(pred, grand, null):
    gap = (grand + null) - torch.sum(pred, dim=1)
    # gap = gap.detach()
    # return pred + gap.unsqueeze(1) / pred.shape[1]
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
def calculate_grand_coalition_U(dataset, batch_size, num_workers, debug, dict_data, dim):
    # print("COMP GRAND COAL")
    grand_key="1"*dim
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    grand = []
    for x in loader:
        x=x[0]
        if debug:
            print("X:", x)
        pred=[]
        for el in x:
            if debug:
                print("EL:", el)    
            DATA_KEY="".join(str(val) for val in el.data.numpy())
            v=dict_data[DATA_KEY][1][grand_key]
            v1, v2 = rearrange_vectors(v)

            v_neg=np.array([v1[0],v2[1]])
            v_pos=np.array([v2[0],v1[1]])

            pred.append([np.abs(v_neg[0]-v_neg[1])/2, np.abs(v_pos[0]-v_pos[1])/2])
        pred= torch.tensor(pred, dtype=torch.float)

        grand.append(pred) 

    # Concatenate and return.
    grand = torch.cat(grand)
    if len(grand.shape) == 1:
        grand = grand.unsqueeze(1)

    return grand

# OK
def generate_validation_data_U(val_set, imputer, validation_samples, sampler, batch_size, num_workers, dict_data, debug):
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
            pred=[]
            for el, mask in zip(x,S):
                if debug:
                    print("EL:", el)
                    print("MASK:", mask)
                mask_key="".join(str(int(val)) for val in mask.data.numpy())
                DATA_KEY="".join(str(val) for val in el.data.numpy())
                v=dict_data[DATA_KEY][1][mask_key]
                v1, v2 = rearrange_vectors(v)
                v_neg=np.array([v1[0],v2[1]])
                v_pos=np.array([v2[0],v1[1]])
                pred.append([np.abs(v_neg[0]-v_neg[1])/2, np.abs(v_pos[0]-v_pos[1])/2])
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
        N = 0
        loss_fn = nn.MSELoss()

        for x, grand, S, values in val_loader:
            # Move to device.
            x = x.to(device)
            S = S.to(device)
            grand = grand.to(device)
            values = values.to(device)

            # Evaluate explainer.
            pred, _ = evaluate_explainer_U(explainer, normalization, x, grand, null, imputer.num_players)

            # Calculate loss.
            approx = null + torch.matmul(S, pred)
            loss = loss_fn(approx, values)

            # Update average.
            N += len(x)
            mean_loss += len(x) * (loss - mean_loss) / N

    return mean_loss


def calculate_grand_inference_U(x, debug, dict_data, dim):
    if debug:
        print("CALCULATE_GRAND_INFERENCE")
        print("X:", x, x[0])
    grand_key="1"*dim
    grand = []
    DATA_KEY="".join(str(val) for val in x[0].data.numpy())
    v=dict_data[DATA_KEY][1][grand_key]
    v1, v2 = rearrange_vectors(v)
    v_neg=np.array([v1[0],v2[1]])
    v_pos=np.array([v2[0],v1[1]])
    grand.append([np.abs(v_neg[0]-v_neg[1])/2, np.abs(v_pos[0]-v_pos[1])/2])

    grand = torch.tensor(grand, dtype=torch.float32)

    return grand


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
              debug=False,
              debug_val=False,
              verbose=False,
              train_dict_data=None,
              val_dict_data=None,
            ):

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
        grand_train = calculate_grand_coalition_U(train_set, batch_size * num_samples, num_workers, False, train_dict_data, self.num_players).cpu()
        grand_val = calculate_grand_coalition_U(val_set, batch_size * num_samples, num_workers, False, val_dict_data, self.num_players).cpu()

        # Null coalition.
        null_key="0"*num_players
        single_sample=train_set[0][0].unsqueeze(0).data.numpy()
        if debug:
            print("SINGLE SAMPLE:", single_sample)
        DATA_KEY="".join(str(val) for val in single_sample[0])
        if debug:
            print("DATA KEY NULL:", DATA_KEY)
        null=train_dict_data[DATA_KEY][1][null_key]
        v1, v2 = rearrange_vectors(null)
        v_neg=np.array([v1[0],v2[1]])
        v_pos=np.array([v2[0],v1[1]])
        null=[np.abs(v_neg[0]-v_neg[1])/2, np.abs(v_pos[0]-v_pos[1])/2]
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
        val_S, val_values = generate_validation_data_U(val_set, imputer, validation_samples, sampler, batch_size * num_samples, num_workers, val_dict_data, False)

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

            for x, grand in batch_iter:
                # Sample S.
                S = sampler.sample(batch_size * num_samples, paired_sampling=paired_sampling)

                # Move to device.
                x = x.to(device)
                S = S.to(device)
                grand = grand.to(device)

                # Evaluate value function.
                x_tiled = x.unsqueeze(1).repeat(
                    1, num_samples, *[1 for _ in range(len(x.shape) - 1)]
                    ).reshape(batch_size * num_samples, *x.shape[1:])
                
                values=[]
                for el, mask in zip(x_tiled,S):
                    mask_key="".join(str(int(val)) for val in mask.data.numpy())
                    DATA_KEY="".join(str(val) for val in el.data.numpy())
                    v=train_dict_data[DATA_KEY][1][mask_key]
                    v1, v2 = rearrange_vectors(v)
                    v_neg=np.array([v1[0],v2[1]])
                    v_pos=np.array([v2[0],v1[1]])
                    values.append([np.abs(v_neg[0]-v_neg[1])/2, np.abs(v_pos[0]+v_pos[1])/2])
                values= torch.tensor(values, dtype=torch.float)
                    # print("VALUES:", values.shape)
                    # print("GRAND:", grand.shape, grand.dtype)

                # Evaluate explainer.
                pred, total = evaluate_explainer_U(explainer, normalization, x, grand, null, num_players)

                # Calculate loss.
                S = S.reshape(batch_size, num_samples, num_players)
                values = values.reshape(batch_size, num_samples, -1)

                approx = null + torch.matmul(S, pred) # SUM

                loss = loss_fn(approx, values)

                loss = loss * num_players
                loss.backward()
                optimizer.step()
                explainer.zero_grad()

            # Evaluate validation loss.
            explainer.eval()
            val_loss = num_players * validate_U(val_loader, imputer, explainer, null, link, normalization).item()
            explainer.train()

            # Save loss, print progress.
            if verbose:
                print('----- Epoch = {} -----'.format(epoch + 1))
                print('Val loss = {:.6f}'.format(val_loss))
                print('')
            scheduler.step(val_loss)
            self.loss_list.append(val_loss)

            # Check for convergence.
            if self.loss_list[-1] < best_loss:
                best_loss = self.loss_list[-1]
                best_epoch = epoch
                best_model = deepcopy(explainer)
                if verbose:
                    print('New best epoch, loss = {:.6f}'.format(val_loss))
                    print('')
            elif epoch - best_epoch == lookback:
                if verbose:
                    print('Stopping early at epoch = {}'.format(epoch))
                break

        # Copy best model.
        for param, best_param in zip(explainer.parameters(), best_model.parameters()):
            param.data = best_param.data
        explainer.eval()

    def shap_values(self, x, dict_mapping, debug):
        
        # Ensure null coalition is calculated.
        device = next(self.explainer.parameters()).device
        if self.null is None:
            null_key="0"*self.num_players
            single_sample=x[:1]
            if debug:
                print("SINGLE SAMPLE:", single_sample)
            DATA_KEY="".join(str(val) for val in single_sample[0].data.numpy())
            if debug:
                print("DATA KEY NULL:", DATA_KEY)
            null=dict_mapping[DATA_KEY][1][null_key]
            v1, v2 = rearrange_vectors(null)
            v_neg=np.array([v1[0],v2[1]])
            v_pos=np.array([v2[0],v1[1]])
            null=[np.abs(v_neg[0]-v_neg[1])/2, np.abs(v_pos[0]-v_pos[1])/2]
            null=torch.tensor(null, dtype=torch.float)
            self.null=null

        # Generate explanations.
        with torch.no_grad():
            # Calculate grand coalition (for normalization).
            grand = calculate_grand_inference_U(x, debug, dict_mapping, self.num_players)

            # Evaluate explainer.
            x = x.to(device)
            pred = evaluate_explainer_U(self.explainer, self.normalization, x, grand, self.null, self.imputer.num_players, inference=True)
            
        pred=pred.cpu().data.numpy()
        
        # r_phi = self.reformulated_phi(pred, x)

        return pred #r_phi