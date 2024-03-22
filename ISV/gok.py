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

def additive_efficient_normalization_MF_2V(pred1, grand1, null1, pred2, grand2, null2,debug, epoch, iter, NE, TO_SWAP):

    # SWAP ELEMENTS
    #if NE<=0.01:
    if TO_SWAP:
        SWAP=False
        pred1_O=pred1.clone()
        pred2_O=pred2.clone()
        index=0
        pred1_new=pred1.clone()
        pred2_new=pred2.clone()
        for isv1, isv2 in zip(pred1, pred2):
            for index2 in range(pred1.shape[1]):
                if isv1[index2][0]>isv2[index2][1]:
                    SWAP=True
                    pred1_new[index][index2][0]=isv2[index2][1]
                    pred2_new[index][index2][1]=isv1[index2][0]
                else:
                    pred1_new[index][index2][0]=isv1[index2][0]
                    pred2_new[index][index2][1]=isv2[index2][1]

                if isv2[index2][0]>isv1[index2][1]:
                    SWAP=True
                    pred1_new[index][index2][1]=isv2[index2][0]
                    pred2_new[index][index2][0]=isv1[index2][1]
                else:
                    pred1_new[index][index2][1]=isv1[index2][1]
                    pred2_new[index][index2][0]=isv2[index2][0]
            index+=1

        pred1=pred1_new
        pred2=pred2_new
        if SWAP==True and torch.equal(pred1_O,pred1):
            print("ERROR")
        if SWAP==True and torch.equal(pred2_O,pred2):
            print("ERROR")

    if debug and epoch==0 and iter<1:
        print("")
        print("### NORMALIZATION ###")
        print("PRED1", pred1.shape)
        print("GRAND1", grand1.shape)
        print("NULL1", null1.shape)
        print("PRED2", pred2.shape)
        print("GRAND2", grand2.shape)
        print("NULL2", null2.shape)

    first_element1 = (grand1 - null1) # CONTROLLO A PRIORI
    second_element1 = torch.sum(pred1, dim=1)

    first_element2 = (grand2 - null2) # CONTROLLO A PRIORI
    second_element2 = torch.sum(pred2, dim=1)

    # first_element1 from tensor to numpy
    first_element1_n=first_element1.detach().numpy()
    first_element2_n=first_element2.detach().numpy()
    second_element1_n=second_element1.detach().numpy()
    second_element2_n=second_element2.detach().numpy()

    if debug and epoch==0 and iter<1:
        print("FIRST_ELEMENT1", first_element1_n)
        print("FIRST_ELEMENT2", first_element2_n)
        print("SECOND_ELEMENT1", second_element1_n)
        print("SECOND_ELEMENT2", second_element2_n)


    fe_lower_neg=first_element1_n[:,0]
    fe_upper_neg=first_element2_n[:,1]
    fe_lower_pos=first_element2_n[:,0]
    fe_upper_pos=first_element1_n[:,1]

    if debug and epoch==0 and iter<1:
        print("FE_LOWER_NEG", fe_lower_neg)
        print("FE_UPPER_NEG", fe_upper_neg)
        print("FE_LOWER_POS", fe_lower_pos)
        print("FE_UPPER_POS", fe_upper_pos)

    se_lower_neg=second_element1_n[:,0]
    se_upper_neg=second_element2_n[:,1]
    se_lower_pos=second_element2_n[:,0]
    se_upper_pos=second_element1_n[:,1]

    if debug and epoch==0 and iter<1:
        print("SE_LOWER_NEG", se_lower_neg)
        print("SE_UPPER_NEG", se_upper_neg)
        print("SE_LOWER_POS", se_lower_pos)
        print("SE_UPPER_POS", se_upper_pos)

    # ------------------------------------------------------
    # CHECK ON |I|>=|J|
    delta_fe_neg=np.abs(fe_upper_neg-fe_lower_neg)
    delta_fe_pos=np.abs(fe_upper_pos-fe_lower_pos)

    delta_se_neg=np.abs(se_upper_neg-se_lower_neg)
    delta_se_pos=np.abs(se_upper_pos-se_lower_pos)

    if debug and epoch==0 and iter<1:
        print("DELTA_FE_NEG", delta_fe_neg)
        print("DELTA_FE_POS", delta_fe_pos)
        print("DELTA_SE_NEG", delta_se_neg)
        print("DELTA_SE_POS", delta_se_pos)

    diff_neg=delta_se_neg-delta_fe_neg
    diff_pos=delta_se_pos-delta_fe_pos

    diff_neg[diff_neg<0]=0
    diff_pos[diff_pos<0]=0

    if debug and epoch==0 and iter<1:
        print("DIFF_NEG", diff_neg)
        print("DIFF_POS", diff_pos)

    #SUB SHIFT PREDICTION
    index=0
    pred1_new=pred1.clone()
    pred2_new=pred2.clone()
    for isv1, isv2 in zip(pred1, pred2):
        if debug and epoch==0 and iter<1:
            print("ELEMENT", index+1, "OF", pred1.shape[0])
            print("DELTA DIFF NEG", diff_neg[index])
            print("DELTA DIFF POS", diff_pos[index])
            print("ISV1", isv1)
            print("ISV2", isv2)
            print("SUM ISV1", torch.sum(isv1, dim=0))
            print("SUM ISV2", torch.sum(isv2, dim=0))
        delta_neg_local=diff_neg[index]/pred1.shape[1]
        delta_pos_local=diff_pos[index]/pred1.shape[1]
        if debug and epoch==0 and iter<1:
            print("DELTA_NEG_LOCAL", delta_neg_local)
            print("DELTA_POS_LOCAL", delta_pos_local)
        for index2 in range(pred1.shape[1]):
            pred1_new[index][index2][0]=isv1[index2][0]+delta_neg_local/2
            pred1_new[index][index2][1]=isv1[index2][1]-delta_pos_local/2
            pred2_new[index][index2][0]=isv2[index2][0]+delta_pos_local/2
            pred2_new[index][index2][1]=isv2[index2][1]-delta_neg_local/2
        if debug and epoch==0 and iter<1:
            print("ISV1_NEW", pred1_new[index])
            print("ISV2_NEW", pred2_new[index])
            print("SUM ISV1_NEW", torch.sum(pred1_new[index], dim=0))
            print("SUM ISV2_NEW", torch.sum(pred2_new[index], dim=0))
        index+=1

    if debug and epoch==0 and iter<0:
        print("PRED1", pred1)
        print("PRED1_NEW", pred1_new)
        print("PRED2", pred2)
        print("PRED2_NEW", pred2_new)
    second_element1 = torch.sum(pred1_new, dim=1)
    second_element2 = torch.sum(pred2_new, dim=1)

    mean_error_delta=np.mean([np.mean(diff_neg), np.mean(diff_pos)])

    count_neg=np.count_nonzero(diff_neg)
    count_pos=np.count_nonzero(diff_pos)
    if debug and epoch==0 and iter<1:
        print("COUNT_NEG", count_neg, len(diff_neg), count_neg/len(diff_neg))
        print("COUNT_POS", count_pos, len(diff_pos), count_pos/len(diff_pos))
    count_neg=count_neg/len(diff_neg)
    count_pos=count_pos/len(diff_pos)
    mean_count_delta=(count_neg+count_pos)/2
    # if debug and epoch==0 and iter<1:
    # if epoch==57:
    #     print("MEAN_COUNT_DELTA", mean_count_delta)

    # ------------------------------------------------------
    # CHECK ON |I|>=|J| POST UPDATE

    second_element1_n=second_element1.detach().numpy()
    second_element2_n=second_element2.detach().numpy()
    
    se_lower_neg=second_element1_n[:,0]
    se_upper_neg=second_element2_n[:,1]
    se_lower_pos=second_element2_n[:,0]
    se_upper_pos=second_element1_n[:,1]

    delta_se_neg=np.abs(se_upper_neg-se_lower_neg)
    delta_se_pos=np.abs(se_upper_pos-se_lower_pos)

    if debug and epoch==0 and iter<1:
        print("DELTA_FE_NEG UP", delta_fe_neg)
        print("DELTA_FE_POS UP", delta_fe_pos)
        print("DELTA_SE_NEG UP", delta_se_neg)
        print("DELTA_SE_POS UP", delta_se_pos)

    diff_neg=delta_se_neg-delta_fe_neg
    diff_pos=delta_se_pos-delta_fe_pos

    diff_neg[diff_neg<0]=0
    diff_pos[diff_pos<0]=0

    if debug and epoch==0 and iter<1:
        print("DIFF_NEG UP", diff_neg)
        print("DIFF_POS UP", diff_pos)

    count_neg=np.count_nonzero(diff_neg)
    count_pos=np.count_nonzero(diff_pos)
    if debug and epoch==0 and iter<1:
        print("COUNT_NEG UP", count_neg, len(diff_neg), count_neg/len(diff_neg))
        print("COUNT_POS UP", count_pos, len(diff_pos), count_pos/len(diff_pos))
    count_neg=count_neg/len(diff_neg)
    count_pos=count_pos/len(diff_pos)
    mean_count_delta=(count_neg+count_pos)/2
    # if debug and epoch==0 and iter<1:
    if epoch==57:
        print("MEAN_COUNT_DELTA UP", mean_count_delta)


    gap1 = first_element1 - second_element1
    pred1=pred1 + gap1.unsqueeze(1) / pred1.shape[1]
    gap2 = first_element2 - second_element2
    pred2=pred2 + gap2.unsqueeze(1) / pred2.shape[1]

    return pred1, pred2, mean_count_delta#/len(pred1)#, err_subtract/len(pred1)

def evaluate_explainer_MF(explainer, normalization, x, grand1, grand2, null1, null2, num_players, debug, epoch, NE, iter=0, inference=False):
    # Evaluate explainer.
    pred1, pred2 = explainer(x)

    # Reshape SHAP values.

    # Tabular.
    image_shape = None
    pred1 = pred1.reshape(len(x), num_players, -1)
    pred2 = pred2.reshape(len(x), num_players, -1)


    pred1_ORIG, pred2_ORIG, _= normalization(pred1, grand1, null1, pred2, grand2, null2, debug,  epoch, iter, NE, TO_SWAP=False)

    # Apply normalization.
    if normalization:
        pred1, pred2, delta_error = normalization(pred1, grand1, null1, pred2, grand2, null2, debug,  epoch, iter, NE, TO_SWAP=True)

    total1 = pred1.sum(dim=1)
    total2 = pred2.sum(dim=1)

    # Reshape for inference.
    if inference:
        return pred1, pred2

    return pred1, total1, pred2, total2, delta_error, pred1_ORIG, pred2_ORIG


def calculate_grand_coalition_MF(dataset, batch_size, num_workers, debug, dict_data, dim):
    if debug:
        print("CALCULATE_GRAND_COALITION")
    #ones = torch.ones(batch_size, imputer.num_players, dtype=torch.float32, device=device)  # 32x12
    grand_key="1"*dim
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    grand1 = []
    grand2 = []
    for x in loader:
        x=x[0]
        if debug:
            print("X:", x)
        for el in x:
            if debug:
                print("EL:", el)    
            DATA_KEY="".join(str(val) for val in el.data.numpy())
            v=dict_data[DATA_KEY][1][grand_key]
            v1, v2 = rearrange_vectors(v)
            grand1.append(v1)
            grand2.append(v2)
            if debug:
                print("Key:", DATA_KEY, "V1:", v1, "V2:", v2)

    grand1 = torch.tensor(grand1, dtype=torch.float32)
    grand2 = torch.tensor(grand2, dtype=torch.float32)

    return grand1, grand2

def calculate_grand_inference_MF(x, debug, dict_data, dim):
    if debug:
        print("CALCULATE_GRAND_INFERENCE")
        print("X:", x, x[0])
    grand_key="1"*dim
    grand1 = []
    grand2 = []
    DATA_KEY="".join(str(val) for val in x[0].data.numpy())
    v=dict_data[DATA_KEY][1][grand_key]
    v1, v2 = rearrange_vectors(v)
    grand1.append(v1)
    grand2.append(v2)

    grand1 = torch.tensor(grand1, dtype=torch.float32)
    grand2 = torch.tensor(grand2, dtype=torch.float32)

    return grand1, grand2


def generate_validation_data_MF(val_set, imputer, validation_samples, sampler, batch_size, num_workers, dict_data, debug):
    # Generate coalitions.
    # print("")
    val_S = sampler.sample(validation_samples * len(val_set), paired_sampling=True).reshape(len(val_set), validation_samples, imputer.num_players)

    # Get values.
    val_values1 = []
    val_values2 = []
    for i in range(validation_samples):
        # Set up data loader.
        dset = DatasetRepeat([val_set, TensorDataset(val_S[:, i])])
        loader = DataLoader(dset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
        values1 = []
        values2 = []

        for x, S in loader:
            for el, mask in zip(x,S):
                if debug:
                    print("EL:", el)
                    print("MASK:", mask)
                mask_key="".join(str(int(val)) for val in mask.data.numpy())
                DATA_KEY="".join(str(val) for val in el.data.numpy())
                v=dict_data[DATA_KEY][1][mask_key]
                v1, v2 = rearrange_vectors(v)
                values1.append(v1)
                values2.append(v2)

        values1 = torch.tensor(values1)
        values2 = torch.tensor(values2)
        if debug:
            print("VALUES1:", values1.shape)
            print("VALUES2:", values2.shape)
        val_values1.append(values1)
        val_values2.append(values2)


    val_values1 = torch.stack(val_values1, dim=1)
    val_values2 = torch.stack(val_values2, dim=1)

    return val_S, val_values1.type(torch.float32), val_values2.type(torch.float32)


def validate_MF(val_loader, imputer, explainer, null1, null2, normalization, approx_null, epoch, debug_val, debug, constraint, alpha):
    if debug_val:
        print("\tVALIDATE")
    with torch.no_grad():
        # Setup.
        device = next(explainer.parameters()).device
        mean_loss = 0
        N = 0
        loss_fn = nn.MSELoss()
        delta_error_V, sub_error_V = 0, 0
        mean_ne_V=0
        mean_ne_ORIG_V=0
        loss_1_V, loss_2_V = 0, 0

        iter_V=0
        for x, grand1, grand2, S, values1, values2 in val_loader: #[val_set, TensorDataset(grand_val1, grand_val2, val_S, val_values1, val_values2)]
            # Move to device.
            x = x.to(device)
            if debug_val and epoch==0 and debug:
                print("VALIDATION x shape", x.shape)
            S = S.to(device)

            grand1 = grand1.to(device)
            grand2 = grand2.to(device)
            values1 = values1.to(device)
            values2 = values2.to(device)

            # Evaluate explainer.
            if epoch==0 and iter_V==0:
                pred1, _, pred2, _, delta_error, pred1_ORIG, pred2_ORIG = evaluate_explainer_MF(explainer, normalization, x, grand1, grand2, null1, null2, imputer.num_players, debug_val, 1, epoch, inference=False)
            else:
                pred1, _, pred2, _, delta_error, pred1_ORIG, pred2_ORIG = evaluate_explainer_MF(explainer, normalization, x, grand1, grand2, null1, null2, imputer.num_players, debug_val, mean_ne_V, epoch, inference=False)


            p1_n=pred1.detach().numpy()
            p2_n=pred2.detach().numpy()

            NE_N=[]
            NE_P=[]
            for el1, el2 in zip(p1_n, p2_n):
                #el1 and el2 6x2
                ne_n=0
                ne_p=0
                for feat1, feat2 in zip(el1, el2):
                    if feat1[0]>feat2[1]:
                        ne_n+=1
                    if feat2[0]>feat1[1]:
                        ne_p+=1
                NE_N.append(ne_n)
                NE_P.append(ne_p)
            ne=np.mean([np.mean(NE_N), np.mean(NE_P)])


            p1_n=pred1_ORIG.detach().numpy()
            p2_n=pred2_ORIG.detach().numpy()

            NE_N=[]
            NE_P=[]
            for el1, el2 in zip(p1_n, p2_n):
                #el1 and el2 6x2
                ne_n=0
                ne_p=0
                for feat1, feat2 in zip(el1, el2):
                    if feat1[0]>feat2[1]:
                        ne_n+=1
                    if feat2[0]>feat1[1]:
                        ne_p+=1
                NE_N.append(ne_n)
                NE_P.append(ne_p)
            ne_ORIG=np.mean([np.mean(NE_N), np.mean(NE_P)])


            if debug_val and epoch==0 and debug:
                print("VALIDATION Sbatch shape", S.shape)
                # print("VALIDATION Sbatch",S[0])
                print("VALIDATION pred1 shape", pred1.shape)
                print("VALIDATION values1 shape", values1.shape)
                print("VALIDATION pred2 shape", pred2.shape)
                print("VALIDATION values2 shape", values2.shape)
            # Calculate loss.
            if approx_null:
                approx1 = null1 + torch.matmul(S, pred1)
                approx2 = null2 + torch.matmul(S, pred2)
            else:
                approx1 = torch.matmul(S, pred1)
                approx2 = torch.matmul(S, pred2)
                
            if debug_val and epoch==0 and debug:
                print("VALIDATION approx shape1", approx1.shape)
                print("VALIDATION approx shape2", approx2.shape)
            loss1 = loss_fn(approx1, values1)
            loss2 = loss_fn(approx2, values2)
            
            # MODIFY
            if constraint:
                temp1 = torch.matmul(S, pred1)
                temp2 = torch.matmul(S, pred2)

                vec1 = temp2[:, 1] - temp1[:, 0]
                vec2 = temp1[:, 1] - temp2[:, 0]
                vec3 = torch.cat((vec1.unsqueeze(1), vec2.unsqueeze(1)), 1)

                temp1 = values1 - null1
                temp2 = values2 - null2

                vec4 = temp2[:, 1] - temp1[:, 0]
                vec5 = temp1[:, 1] - temp2[:, 0]
                vec6 = torch.cat((vec4.unsqueeze(1), vec5.unsqueeze(1)), 1)

                loss3 = loss_fn(vec3, vec6)
                loss = ((1 - alpha) * (loss1 + loss2) + (alpha) * (loss3) + ne_ORIG/len(p1_n) + delta_error)
            else:
                # if mean_ne_V<=0.01:
                #     loss = loss1 + loss2 + ne/len(p1_n) + delta_error
                # else:
                loss = loss1 + loss2 + ne_ORIG/len(p1_n) + delta_error# + sub_error

            # Update average.
            N += len(x)
            loss_1_V += len(x) * (loss1 - loss_1_V) / N
            loss_2_V += len(x) * (loss2 - loss_2_V) / N
            mean_ne_V += len(x) * (ne/len(p1_n) - mean_ne_V) / N
            mean_ne_ORIG_V += len(x) * (ne_ORIG/len(p1_n) - mean_ne_ORIG_V) / N
            delta_error_V += len(x) * (delta_error - delta_error_V) / N
            #sub_error_V += len(x) * (sub_error - sub_error_V) / N
            mean_loss += len(x) * (loss - mean_loss) / N
            iter_V+=1

    if debug_val:
        print("VALIDATION LOSS:", mean_loss)
        print("VALIDATION LOSS1:", loss_1_V)
        print("VALIDATION LOSS2:", loss_2_V)
        print("VALIDATION NE:", mean_ne_V)
        print("VALIDATION NE_ORIG:", mean_ne_ORIG_V)
        print("VALIDATION DELTA_ERROR:", delta_error_V)
    #print("VALIDATION SUB_ERROR:", sub_error_V)

    return mean_loss


class MultiFastSHAP:
    def __init__(
            self,
                explainer,
                imputer,
                normalization='none',
                link=None
                ):
        # Set up explainer, imputer and link function.
        self.explainer = explainer
        self.imputer = imputer
        self.num_players = imputer.num_players
        self.null1 = None
        self.null2 = None
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
            self.normalization = additive_efficient_normalization_MF_2V  # ONE USED
        else:
            raise ValueError('unsupported normalization: {}'.format(normalization))

    def train(
            self,
            train_data,
            val_data,
            batch_size,
            num_samples,
            max_epochs,
            lr=2e-4,
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
            verbose=False,
            weight_decay=0.01,
            approx_null=True,
            debug=False,
            debug_val=False,
            constraint = False,
            alpha = 0.5,
            train_dict_data=None,
            val_dict_data=None,
            ):

        # Set up explainer model.
        explainer = self.explainer  # NEURAL NETWORK
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

        grand_train1, grand_train2 = calculate_grand_coalition_MF(train_set, batch_size * num_samples, num_workers, False, train_dict_data, self.num_players)
        grand_val1, grand_val2 = calculate_grand_coalition_MF(val_set, batch_size * num_samples, num_workers, False, val_dict_data, self.num_players)
        if debug:
            print("GRAND TRAIN1:", grand_train1.shape)#, grand_train1)
            print("GRAND TRAIN2:", grand_train2.shape)#, grand_train2)
            print("GRAND VAL1:", grand_val1.shape)#, grand_val1)
            print("GRAND VAL2:", grand_val2.shape)#, grand_val2)

        # Null coalition.
        null_key="0"*self.num_players
        single_sample=train_set[0][0].unsqueeze(0).data.numpy()
        if debug:
            print("SINGLE SAMPLE:", single_sample)
        DATA_KEY="".join(str(val) for val in single_sample[0])
        if debug:
            print("DATA KEY NULL:", DATA_KEY)
        null=train_dict_data[DATA_KEY][1][null_key]
        null1, null2 = rearrange_vectors(null)
        # convert to tensor
        self.null1 = torch.tensor(null1, dtype=torch.float32)
        self.null2 = torch.tensor(null2, dtype=torch.float32)

        if debug:
            print("NULL1:", self.null1.shape, self.null1)
            print("NULL2:", self.null2.shape, self.null2)

        # Set up train loader.
        train_set_tmp = DatasetRepeat([train_set, TensorDataset(grand_train1, grand_train2)])  # PERMETTE DI AVERE ELEMENTI RIPETUTI QUANDO LA LUN E' DIVERSA
        train_loader = DataLoader(train_set_tmp, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=num_workers) # SHUFFLE
       
        sampler = ShapleySampler(num_players)

        if validation_seed is not None:
            torch.manual_seed(validation_seed)
        val_S, val_values1, val_values2 = generate_validation_data_MF(val_set, imputer, validation_samples, sampler, batch_size * num_samples, num_workers, val_dict_data, False)
        if debug:
            print("VAL_S:", val_S.shape)#, val_S)
            print("VAL_VALUES1:", val_values1.shape)#, val_values1)
            print("VAL_VALUES2:", val_values2.shape)#, val_values2)

        # Set up val loader.
        v_ds=TensorDataset(grand_val1, grand_val2, val_S, val_values1, val_values2)
        val_set_tmp = DatasetRepeat([val_set, v_ds])
        val_loader = DataLoader(val_set_tmp, batch_size=batch_size * num_samples, pin_memory=True, num_workers=num_workers)

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

            mean_ne_T=0
            mean_ne_ORIG_T=0
            Nt=0
            delta_error_T, sub_error_T = 0, 0
            loss_T = 0
            loss_1_T, loss_2_T = 0, 0

            for iter, v2 in enumerate(batch_iter):
                x, grand1, grand2 = v2
                # Sample S.
                if debug and epoch == 0 and iter < 5:
                    print("DATA:", x.shape)#, x)
                    print("GRAND1:", grand1.shape)#, grand1)
                    print("GRAND2:", grand2.shape)#, grand2)

                S = sampler.sample(batch_size * num_samples, paired_sampling=paired_sampling)
                if debug and epoch == 0 and iter == 0:
                    print("SUBSET:", S.shape)#, S)

                # Move to device.
                x = x.to(device)
                S = S.to(device)
                grand1 = grand1.to(device)
                grand2 = grand2.to(device)

                # Evaluate value function.
                if debug and epoch == 0 and iter < 5:
                    print("UNSQUEEZE:", x.unsqueeze(1).shape)
                
                x_tiled = x.unsqueeze(1).repeat(
                    1, num_samples, *[1 for _ in range(len(x.shape) - 1)]
                ).reshape(batch_size * num_samples, *x.shape[1:])
                if debug and epoch == 0 and iter < 5:
                    print("X_TILED", x_tiled.shape)#, x_tiled)
                    
                values1 = []
                values2 = []
                for el, mask in zip(x_tiled,S):
                    mask_key="".join(str(int(val)) for val in mask.data.numpy())
                    DATA_KEY="".join(str(val) for val in el.data.numpy())
                    v=train_dict_data[DATA_KEY][1][mask_key]
                    v1, v2 = rearrange_vectors(v)
                    values1.append(v1)
                    values2.append(v2)
                values1 = torch.tensor(values1, dtype=torch.float32)
                values2 = torch.tensor(values2, dtype=torch.float32)

                if debug and epoch == 0 and iter < 5:
                    print("VALUES1", values1.shape)#, values1)
                    print("VALUES2", values2.shape)#, values2)

  
                if epoch==0 and iter==0:
                    pred1, total1, pred2, total2, delta_error, pred1_ORIG, pred2_ORIG = evaluate_explainer_MF(explainer, normalization, x, grand1, grand2, null1, null2, num_players, debug, 1, epoch, iter)  # NULL PER LA NORMALIZZAZIONE
                else:
                    pred1, total1, pred2, total2, delta_error, pred1_ORIG, pred2_ORIG = evaluate_explainer_MF(explainer, normalization, x, grand1, grand2, null1, null2, num_players, debug, mean_ne_T, epoch, iter)  # NULL PER LA NORMALIZZAZIONE


                if debug and epoch == 0 and iter < 5:
                    print("PRED1:", pred1.shape, pred1)
                    print("TOTAL1:", total1.shape)
                    print("PRED2:", pred2.shape, pred2)
                    print("TOTAL2:", total2.shape)

                p1_n=pred1.detach().numpy()
                p2_n=pred2.detach().numpy()
                
                NE_N=[]
                NE_P=[]
                for el1, el2 in zip(p1_n, p2_n):
                    #el1 and el2 6x2
                    ne_n=0
                    ne_p=0
                    for feat1, feat2 in zip(el1, el2):
                        if feat1[0]>feat2[1]:
                            ne_n+=1
                        if feat2[0]>feat1[1]:
                            ne_p+=1
                    NE_N.append(ne_n)
                    NE_P.append(ne_p)
                ne=np.mean([np.mean(NE_N), np.mean(NE_P)])
                if debug and epoch == 0 and iter < 5:
                    print("NE:", ne)
                    print("NE_N", NE_N)
                    print("NE_P", NE_P)
                    print("LEN P1_N:", len(p1_n))


                p1_n=pred1_ORIG.detach().numpy()
                p2_n=pred2_ORIG.detach().numpy()

                NE_N=[]
                NE_P=[]
                for el1, el2 in zip(p1_n, p2_n):
                    #el1 and el2 6x2
                    ne_n=0
                    ne_p=0
                    for feat1, feat2 in zip(el1, el2):
                        if feat1[0]>feat2[1]:
                            ne_n+=1
                        if feat2[0]>feat1[1]:
                            ne_p+=1
                    NE_N.append(ne_n)
                    NE_P.append(ne_p)
                ne_ORIG=np.mean([np.mean(NE_N), np.mean(NE_P)])
                

                # Calculate loss.
                S = S.reshape(batch_size, num_samples, num_players)
                if debug and epoch == 0 and iter < 5:
                    print("S RESHAPE", S.shape)

                # print("value shape", values.shape)
                values1 = values1.reshape(batch_size, num_samples, -1)
                values2 = values2.reshape(batch_size, num_samples, -1)
                if debug and epoch == 0 and iter < 5:
                    print("VALUES RESHAPE1", values1.shape, values1)
                    print("VALUES RESHAPE2", values2.shape, values2)

                # print(type(S), type(pred1), type(pred2), type(null1), type(null2))
                if approx_null:
                    approx1 = self.null1 + torch.matmul(S, pred1)
                    approx2 = self.null2 + torch.matmul(S, pred2)
                else:
                    approx1 = torch.matmul(S, pred1)
                    approx2 = torch.matmul(S, pred2)

                if debug and epoch == 0 and iter < 5:
                    print("APPROX1", approx1.shape, approx1)
                    print("APPROX2", approx2.shape, approx2)

                loss1 = loss_fn(approx1, values1)
                loss2 = loss_fn(approx2, values2)
                if eff_lambda:
                    print("EFF_LAMBDA")
                    #loss = loss + eff_lambda * loss_fn(total, grand - null)

                # MODIFY
                if constraint: 
                    temp1 = torch.matmul(S, pred1)
                    temp2 = torch.matmul(S, pred2)

                    vec1 = temp2[:, 1] - temp1[:, 0] #delta_neg
                    vec2 = temp1[:, 1] - temp2[:, 0] #delta_pos
                    vec3 = torch.cat((vec1.unsqueeze(1), vec2.unsqueeze(1)), 1)

                    temp1=values1-null1
                    temp2=values2-null2

                    vec4 = temp2[:, 1] - temp1[:, 0] #delta_neg
                    vec5 = temp1[:, 1] - temp2[:, 0] #delta_pos
                    vec6 = torch.cat((vec4.unsqueeze(1), vec5.unsqueeze(1)), 1)

                    loss3 = loss_fn(vec3, vec6)
                    loss = num_players * ((1-alpha)*(loss1 + loss2) + (alpha)*(loss3) + ne_ORIG/len(p1_n) + delta_error)
                else:
                    loss = num_players * (loss1 + loss2 + ne_ORIG/len(p1_n) + delta_error) # + sub_error)

                Nt+=len(x)
                loss_T += len(x) * (loss - loss_T) / Nt
                loss_1_T += len(x) * (loss1 - loss_1_T) / Nt
                loss_2_T += len(x) * (loss2 - loss_2_T) / Nt
                mean_ne_T += len(x) * (ne/len(p1_n) - mean_ne_T) / Nt
                mean_ne_ORIG_T += len(x) * (ne_ORIG/len(p1_n) - mean_ne_ORIG_T) / Nt
                delta_error_T += len(x) * (delta_error - delta_error_T) / Nt



                loss.backward()
                optimizer.step()
                explainer.zero_grad()

            if verbose:
                print('----- Epoch = {} -----'.format(epoch + 1))
                print('LOSS = {:.6f}'.format(loss_T))
                print('NE = {:.6f}'.format(mean_ne_T))
                print('NE_ORIG = {:.6f}'.format(mean_ne_ORIG_T))
                print('DELTA_ERROR = {:.6f}'.format(delta_error_T))
                # print('SUB_ERROR = {:.6f}'.format(sub_error_T))
                print('LOSS1 = {:.6f}'.format(loss_1_T))
                print('LOSS2 = {:.6f}'.format(loss_2_T))


            # Evaluate validation loss.
            explainer.eval()
            val_loss = num_players * validate_MF(val_loader, imputer, explainer, self.null1, self.null2, normalization, approx_null, epoch, debug_val=debug_val, debug=debug, constraint=constraint, alpha=alpha)  # .item()
            explainer.train()

            scheduler.step(val_loss)
            self.loss_list.append(val_loss)
            if verbose:
                # print('----- Epoch = {} -----'.format(epoch + 1))
                print('\nTraining loss = {:.6f}'.format(loss_T))
                print('Validation loss = {:.6f}'.format(val_loss))

            # Check for convergence.
            if self.loss_list[-1] < best_loss:
                best_loss = self.loss_list[-1]
                best_epoch = epoch
                best_model = deepcopy(explainer)
                if verbose:
                    # print('----- Epoch = {} -----'.format(epoch + 1))
                    print('\tNew best epoch, loss = {:.6f}'.format(val_loss))
                    print('')
            elif epoch - best_epoch == lookback:
                if verbose:
                    print('\tStopping early at epoch = {}'.format(epoch))
                break

        # Copy best model.
        for param, best_param in zip(explainer.parameters(), best_model.parameters()):
            param.data = best_param.data
        explainer.eval()

    def shap_values(self, x, dict_mapping, debug):
        # Data conversion.
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        elif isinstance(x, torch.Tensor):
            pass
        else:
            raise ValueError('data must be np.ndarray or torch.Tensor')
            
        device = next(self.explainer.parameters()).device
            
        if self.null1 is None or self.null2 is None:
            if debug:
                print("NULL INITIALIZATION")
            
            # Null coalition.
            null_key="0"*self.num_players
            single_sample=x[:1]
            if debug:
                print("SINGLE SAMPLE:", single_sample)
            DATA_KEY="".join(str(val) for val in single_sample[0].data.numpy())
            if debug:
                print("DATA KEY NULL:", DATA_KEY)
            null=dict_mapping[DATA_KEY][1][null_key]
            null1, null2 = rearrange_vectors(null)
            # convert to tensor
            self.null1 = torch.tensor(null1, dtype=torch.float32)
            self.null2 = torch.tensor(null2, dtype=torch.float32)
        

        # Generate explanations.
        with torch.no_grad():
            grand1, grand2 = calculate_grand_inference_MF(x, debug, dict_mapping, self.num_players)  # CALCOLO CON TUTTI A 1

            # Evaluate explainer.
            x = x.to(device)
            
            pred1, pred2 = evaluate_explainer_MF(self.explainer, self.normalization, x, grand1, grand2, self.null1, self.null2, self.num_players, debug, 0, 0, inference=True)  # NULL PER LA NORMALIZZAZIONE

        return pred1.cpu().data.numpy(), pred2.cpu().data.numpy()