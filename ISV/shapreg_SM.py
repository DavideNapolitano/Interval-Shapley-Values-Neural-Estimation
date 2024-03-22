import warnings
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain, combinations

import warnings
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm.auto import tqdm
import sys

sys.path.append('IntervalSV/code')

from utils import *

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

###########################################################################    FOCUS
class ShapleyValues:
    '''For storing and plotting Shapley values.'''
    def __init__(self, values, std):
        self.values = values
        self.std = std

    def save(self, filename):
        '''Save Shapley values object.'''
        if isinstance(filename, str):
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise TypeError('filename must be str')

    def __repr__(self):
        with np.printoptions(precision=2, threshold=12, floatmode='fixed'):
            return 'Shapley Values(\n  (Mean): {}\n  (Std):  {}\n)'.format(self.values, self.std)
    

class CooperativeGame:
    '''Base class for cooperative games.'''

    def __init__(self):
        raise NotImplementedError

    def __call__(self, S):
        '''Evaluate cooperative game.'''
        raise NotImplementedError

    def grand(self):
        '''Get grand coalition value.'''
        return self.__call__(np.ones((1, self.players), dtype=bool))[0]

    def null(self):
        '''Get null coalition value.'''
        return self.__call__(np.zeros((1, self.players), dtype=bool))[0]

    
class PredictionGame(CooperativeGame):

    def __init__(self, extension, sample, groups=None):
        # Add batch dimension to sample.
        if sample.ndim == 1:
            sample = sample[np.newaxis]
        elif sample.shape[0] != 1:
            raise ValueError('sample must have shape (ndim,) or (1,ndim)')

        self.extension = extension
        self.sample = sample

        # Store feature groups.
        num_features = sample.shape[1]

        self.players = num_features
        self.groups_matrix = None

        # Caching.
        self.sample_repeat = sample

    def __call__(self, S):

        # Try to use caching for repeated data.
        if len(S) != len(self.sample_repeat):
            self.sample_repeat = self.sample.repeat(len(S), 0)
        input_data = self.sample_repeat

        # Apply group transformation.
        if self.groups_matrix is not None:
            S = np.matmul(S, self.groups_matrix)

        # Evaluate.
        return self.extension(input_data, S)

def default_min_variance_samples(game):
    '''Determine min_variance_samples.'''
    return 5

def default_variance_batches(game, batch_size):

    if isinstance(game, CooperativeGame):
        return int(np.ceil(10 * game.players / batch_size))
    else:
        # Require more intermediate samples for stochastic games.
        return int(np.ceil(25 * game.players / batch_size))

###########################################################################    FOCUS
def calculate_result(A, b, total):
    num_players = A.shape[1]
    try:
        if len(b.shape) == 2:
            A_inv_one = np.linalg.solve(A, np.ones((num_players, 1)))
        else:
            A_inv_one = np.linalg.solve(A, np.ones(num_players))
            
        A_inv_vec = np.linalg.solve(A, b)
        
        values = (A_inv_vec - A_inv_one * (np.sum(A_inv_vec, axis=0, keepdims=True) - total) / np.sum(A_inv_one)) ################   FOCUS
    except np.linalg.LinAlgError:
        raise ValueError('singular matrix inversion. Consider using larger variance_batches')

    return values

def computer(MA,MB,MT):
    num_players = MA.shape[1]
    try:
        if len(MB.shape) == 2:
            A_inv_one = np.linalg.solve(MA, np.ones((num_players, 1)))
        else:
            A_inv_one = np.linalg.solve(MA, np.ones(num_players))
            
        A_inv_vec = np.linalg.solve(MA, MB)
        # print("A_inv_vec_1", A_inv_vec_1.shape)
        
        TERM1=np.sum(A_inv_vec, axis=0, keepdims=True)
        TERM2=A_inv_one * (TERM1 - MT) / np.sum(A_inv_one)
        
        # print("TERM1_1", TERM1_1.shape, TERM1_1) 
        # print("TERM2_1", TERM2_1.shape)
        return TERM1, TERM2, A_inv_vec
    except np.linalg.LinAlgError:
        raise ValueError('singular matrix inversion. Consider using larger variance_batches')

    
def checker(b_1_tmp, b_2_tmp, A_tmp, total_1, total_2):
    TERM1_1, TERM2_1, A_inv_vec_1 = computer(A_tmp, b_1_tmp, total_1)
    TERM1_2, TERM2_2, A_inv_vec_2 = computer(A_tmp, b_2_tmp, total_2)
    
    # TO CHECK
    delta_FE_NEG = np.abs(TERM1_2[0][0] - TERM1_1[0][0])
    delta_FE_POS = np.abs(TERM1_2[0][1] - TERM1_1[0][1])
    
    delta_SE_NEG = np.abs(total_2[0] - total_1[0])
    delta_SE_POS = np.abs(total_2[1] - total_1[1])
    
    if delta_FE_NEG >= delta_SE_NEG and delta_FE_POS >= delta_SE_POS:
        # print("\tFIRST SUBSTACTION IS VALID")
        d_NEG = np.abs(A_inv_vec_2[:,0]-A_inv_vec_1[:,0])
        d_POS = np.abs(A_inv_vec_2[:,1]-A_inv_vec_1[:,1])

        d_T2_NEG = np.abs(TERM2_2[:,0]-TERM2_1[:,0])
        d_T2_POS = np.abs(TERM2_2[:,1]-TERM2_1[:,1])

        diff_1 =  d_T2_NEG - d_NEG
        diff_2 =  d_T2_POS - d_POS

        diff_1[diff_1 < 0] = 0
        diff_2[diff_2 < 0] = 0

        count_1 = np.count_nonzero(diff_1)
        count_2 = np.count_nonzero(diff_2)

        return count_1, count_2, A_inv_vec_1, A_inv_vec_2, TERM2_1, TERM2_2
    else:
        return -1, -1, A_inv_vec_1, A_inv_vec_2, TERM2_1, TERM2_2

###########################################################################    FOCUS
def ShapleyRegression2(game_1,
                      game_2,
                      batch_size=512,
                      detect_convergence=True,
                      thresh=0.01,
                      n_samples=None,
                      paired_sampling=False,
                      min_variance_samples=None,
                      variance_batches=None,
                      bar=True,
                      verbose=False,
                      return_all=False,
                      num_features=None
                    ):
    
    min_variance_samples_1 = default_min_variance_samples(game_1)
    variance_batches_1 = default_variance_batches(game_1, batch_size)
    min_variance_samples_2 = default_min_variance_samples(game_2)
    variance_batches_2 = default_variance_batches(game_2, batch_size)

    n_samples = 1e20
    if not detect_convergence:
        detect_convergence = True
        if verbose:
            print('Turning convergence detection on')

    if detect_convergence:
        assert 0 < thresh < 1

    # Weighting kernel (probability of each subset size).
    num_players = game_1.players
    weights = np.arange(1, num_players)
    weights = 1 / (weights * (num_players - weights))
    weights = weights / np.sum(weights)
    # print("weights", weights.shape, weights)

    # Calculate null and grand coalitions for constraints.

    null_1 = game_1.null()
    null_2 = game_2.null()
    grand_1 = game_1.grand()
    grand_2 = game_2.grand()

    # Calculate difference between grand and null coalitions.
    total_1 = grand_1 - null_1   
    total_2 = grand_2 - null_2
    # print("total_1", total_1.shape, total_1)
    # print("total_2", total_2.shape, total_2)

    # Set up bar.
    n_loops = int(np.ceil(n_samples / batch_size))
    if bar:
        if detect_convergence:
            bar = tqdm(total=1)
        else:
            bar = tqdm(total=n_loops * batch_size)

    # Setup.
    n = 0
    b_1 = 0
    b_2 = 0
    A = 0
    n_tmp = 0
    b_1_tmp = 0
    b_2_tmp = 0
    A_tmp = 0
    estimate_list_1 = []
    estimate_list_2 = []

    # For variance estimation.
    A_sample_list = []
    b_sample_list_1 = []
    b_sample_list_2 = []

    # For tracking progress.
    var_1 = np.nan * np.ones(num_players)
    var_2 = np.nan * np.ones(num_players)
    if return_all:
        N_list = []
        std_1_list = []
        val_1_list = []
        std_2_list = []
        val_2_list = []

    all_subsets=list(powerset(np.arange(num_features)))#[:-1]
    mask=[]
    for el in all_subsets:
        tmp=np.zeros(num_features, dtype=bool)
        np.put(tmp, list(el), np.ones(len(el), dtype=bool))
        mask.append(tmp)
    mask=np.array(mask)
    # print("mask", mask.shape)

    ratio_1=np.inf
    ratio_2=np.inf

    values1=None
    values2=None

    sampler=ShapleySampler(num_players)

    # Begin sampling.
    REAL_IT=0
    for it in range(n_loops):
        # print("ITERATION", it, "RATIO 1", ratio_1, "RATIO 2", ratio_2)

        if values1 is not None and values2 is not None:
            good_S=[]
            for i in range(int(len(mask)/batch_size)):
                if (i+1)*batch_size>len(mask):
                    S=mask[i*batch_size:]
                else:
                    S=mask[i*batch_size:(i+1)*batch_size]
                # print("S", S.shape)

                b_sample_1 = (S.astype(float).T * (game_1(S) - null_1)[:, np.newaxis].T).T 
                b_sample_2 = (S.astype(float).T * (game_2(S) - null_2)[:, np.newaxis].T).T  

                # print("b_sample_1", b_sample_1.shape)
                # print("b_sample_2", b_sample_2.shape)

                # print('CHECK INTERVAL ON B')
                current_delta_neg = b_2[0, :] - b_1[0, :]
                current_delta_pos = b_2[1, :] - b_1[1, :]

                for idx in range(len(b_sample_1)):
                    b_smp_1 = b_sample_1[idx]
                    b_smp_2 = b_sample_2[idx]
                    
                    delta_neg = np.abs(b_smp_2[0, :] - b_smp_1[0, :])
                    delta_pos = np.abs(b_smp_2[1, :] - b_smp_1[1, :])
                    
                    diff_neg = current_delta_neg - delta_neg
                    diff_pos = current_delta_pos - delta_pos 
                    
                    diff_neg[diff_neg < 0] = 0
                    diff_pos[diff_pos < 0] = 0
                    
                    count_neg = np.count_nonzero(diff_neg)
                    count_pos = np.count_nonzero(diff_pos)
                    
                    if count_neg == 0 and count_pos == 0:
                        good_S.append(S[idx])
            good_S=np.array(good_S)
            # print("\tITER", it, "GOOD SAMPLES", len(good_S), "OUT OF", len(mask))

            if len(good_S) < batch_size:
                # repeat elements in good_S to reach batch_size
                good_S = np.repeat(good_S, np.ceil(batch_size/len(good_S)), axis=0)
                good_S = good_S[:batch_size]
                # print("\t\tNEW SIZE REPEATED", len(good_S))

        S=sampler.sample(batch_size=batch_size, paired_sampling=False)
        S=S.cpu().numpy()
        S=S.astype(bool)
        # S=sampler.sample(batch_size=batch_size, paired_sampling=True)
        # S=S.cpu().numpy()
        # S=S.astype(bool)

        # Single sample.
        A_sample = np.matmul(S[:, :, np.newaxis].astype(float),  S[:, np.newaxis, :].astype(float))

        b_sample_1 = (S.astype(float).T * (game_1(S) - null_1)[:, np.newaxis].T).T  ###########################################################################   ALWAYS TRUE |S|>|0|
        b_sample_2 = (S.astype(float).T * (game_2(S) - null_2)[:, np.newaxis].T).T  ###########################################################################   ALWAYS TRUE |S|>|0|
        
        n_tmp=n
        b_1_tmp=b_1
        b_2_tmp=b_2
        A_tmp=A

        n_tmp += batch_size #UPDATE WITH NUMBER OF SAMPLES GOOD
        b_1_tmp += np.sum(b_sample_1 - b_1_tmp, axis=0) / n_tmp     ###########################################################################   POSSIBLE ISSUE
        b_2_tmp += np.sum(b_sample_2 - b_2_tmp, axis=0) / n_tmp     ###########################################################################   POSSIBLE ISSUE
        A_tmp += np.sum(A_sample - A_tmp, axis=0) / n_tmp
        
        count_1, count_2, A_inv_vec_1, A_inv_vec_2, TERM2_1, TERM2_2 = checker(b_1_tmp, b_2_tmp, A_tmp, total_1, total_2)

        if count_1 == 0 and count_2 == 0:

            REAL_IT+=1

            values1 = (A_inv_vec_1 - TERM2_1) ################   FOCUS
            values2 = (A_inv_vec_2 - TERM2_2) ################   FOCUS

            n=n_tmp
            b_1=b_1_tmp
            b_2=b_2_tmp
            A=A_tmp

            A_sample_list.append(A_sample)
            b_sample_list_1.append(b_sample_1)
            b_sample_list_2.append(b_sample_2)

            if len(A_sample_list) == variance_batches_1 and len(A_sample_list) == variance_batches_2:

                # Aggregate samples for intermediate estimate.
                A_sample = np.concatenate(A_sample_list, axis=0).mean(axis=0)
                b_sample_1 = np.concatenate(b_sample_list_1, axis=0).mean(axis=0)
                b_sample_2 = np.concatenate(b_sample_list_2, axis=0).mean(axis=0)
                A_sample_list = []
                b_sample_list_1 = []
                b_sample_list_2 = []

                # Add new estimate.
                count_1, count_2, A_inv_vec_1, A_inv_vec_2, TERM2_1, TERM2_2 = checker(b_sample_1, b_sample_2, A_sample, total_1, total_2)

                if count_1 == 0 and count_2 == 0:
                    # print("\t\tUPDATING VAR")
                    VAL1 = (A_inv_vec_1 - TERM2_1) ################   FOCUS
                    VAL2 = (A_inv_vec_2 - TERM2_2) ################   FOCUS

                    estimate_list_1.append(VAL1)
                    estimate_list_2.append(VAL2)

                    # Estimate current var.
                    if len(estimate_list_1) >= min_variance_samples_1:
                        var_1 = np.array(estimate_list_1).var(axis=0)

                    if len(estimate_list_2) >= min_variance_samples_2:
                        var_2 = np.array(estimate_list_2).var(axis=0)

            # Convergence ratio.

            std_1 = np.sqrt(var_1 * variance_batches_1 / (it + 1))
            ratio_1 = np.max(np.max(std_1, axis=0) / (values1.max(axis=0) - values1.min(axis=0)))

            std_2 = np.sqrt(var_2 * variance_batches_2 / (it + 1))
            ratio_2 = np.max(np.max(std_2, axis=0) / (values2.max(axis=0) - values2.min(axis=0)))

            # Print progress message.
            if verbose:
                if detect_convergence:
                    print(f'StdDev Ratio = {ratio:.4f} (Converge at {thresh:.4f})')
                else:
                    print(f'StdDev Ratio = {ratio:.4f}')

            # Check for convergence.
            if detect_convergence:
                # print("RATIO 1", ratio_1, "RATIO 2", ratio_2)
                if ratio_1 < thresh and ratio_2 < thresh:
                    if verbose:
                        print('Detected convergence')

                    # Skip bar ahead.
                    if bar:
                        bar.n = bar.total
                        bar.refresh()
                    break

            # Forecast number of iterations required.
            if detect_convergence:
                N_est = (it + 1) * (ratio_1 / thresh) ** 2
                if bar and not np.isnan(N_est):
                    bar.n = np.around((it + 1) / N_est, 4)
                    bar.refresh()
            elif bar:
                bar.update(batch_size)

            if return_all:
                val_1_list.append(values1)
                std_1_list.append(std_1)
                val_2_list.append(values2)
                std_2_list.append(std_2)
                if detect_convergence:
                    N_list.append(N_est)

    if return_all:
        # Dictionary for progress tracking.
        iters = (
            (np.arange(it + 1) + 1) * batch_size *
            (1 + int(paired_sampling)))
        tracking_dict = {
            'values_1': val_1_list,
            'std_1': std_1_list,
            'values_2': val_2_list,
            'std_2': std_2_list,
            'iters': iters}
        if detect_convergence:
            tracking_dict['N_est'] = N_list

        return ShapleyValues(values1, std_1), ShapleyValues(values2, std_2), tracking_dict    
    else:
        return ShapleyValues(values1, std_1), ShapleyValues(values2, std_2)


def ShapleyRegression3(game_1,
                      game_2,
                      batch_size=512,
                      detect_convergence=True,
                      thresh=0.01,
                      n_samples=None,
                      paired_sampling=True,
                      return_all=False,
                      min_variance_samples=None,
                      variance_batches=None,
                      bar=True,
                      verbose=False):
    

    min_variance_samples_1 = default_min_variance_samples(game_1)
    variance_batches_1 = default_variance_batches(game_1, batch_size)
    min_variance_samples_2 = default_min_variance_samples(game_2)
    variance_batches_2 = default_variance_batches(game_2, batch_size)

    n_samples = 1e20
    if not detect_convergence:
        detect_convergence = True
        if verbose:
            print('Turning convergence detection on')

    if detect_convergence:
        assert 0 < thresh < 1

    # Weighting kernel (probability of each subset size).
    num_players = game_1.players
    weights = np.arange(1, num_players)
    weights = 1 / (weights * (num_players - weights))
    weights = weights / np.sum(weights)

    # Calculate null and grand coalitions for constraints.

    null_1 = game_1.null()
    null_2 = game_2.null()
    grand_1 = game_1.grand()
    grand_2 = game_2.grand()

    # Calculate difference between grand and null coalitions.
    total_1 = grand_1 - null_1   
    total_2 = grand_2 - null_2
    print("total_1", total_1.shape, total_1)
    print("total_2", total_2.shape, total_2)

    # Set up bar.
    n_loops = int(np.ceil(n_samples / batch_size))
    if bar:
        if detect_convergence:
            bar = tqdm(total=1)
        else:
            bar = tqdm(total=n_loops * batch_size)

    # Setup.
    n = 0
    b_1 = 0
    b_2 = 0
    A = 0
    estimate_list_1 = []
    estimate_list_2 = []

    # For variance estimation.
    A_sample_list = []
    b_sample_list_1 = []
    b_sample_list_2 = []

    # For tracking progress.
    var_1 = np.nan * np.ones(num_players)
    var_2 = np.nan * np.ones(num_players)

    # Begin sampling.
    for it in range(n_loops):
        # print("ITERATION", it, "/", n_loops)
        # Sample subsets.
        S = np.zeros((batch_size, num_players), dtype=bool)
        num_included = np.random.choice(num_players - 1, size=batch_size, p=weights) + 1
        for row, num in zip(S, num_included):
            inds = np.random.choice(num_players, size=num, replace=False)
            row[inds] = 1


        # Single sample.
        A_sample = np.matmul(S[:, :, np.newaxis].astype(float),  S[:, np.newaxis, :].astype(float))

        b_sample_1 = (S.astype(float).T * (game_1(S) - null_1)[:, np.newaxis].T).T  ###########################################################################   ALWAYS TRUE |S|>|0|
        b_sample_2 = (S.astype(float).T * (game_2(S) - null_2)[:, np.newaxis].T).T  ###########################################################################   ALWAYS TRUE |S|>|0|
        

        # print("FOUND GOOD BATCHES", batch_good,"/", batch_size)
        n += batch_size #UPDATE WITH NUMBER OF SAMPLES GOOD
        b_1 += np.sum(b_sample_1 - b_1, axis=0) / n    ###########################################################################   POSSIBLE ISSUE
        b_2 += np.sum(b_sample_2 - b_2, axis=0) / n     ###########################################################################   POSSIBLE ISSUE
        A += np.sum(A_sample - A, axis=0) / n

        # print("A", A.shape)
        # print("b_1", b_1.shape)
        # print("b_2", b_2.shape)

        values1 = calculate_result(A, b_1, total_1)
        values2 = calculate_result(A, b_2, total_2)

        A_sample_list.append(A_sample)
        b_sample_list_1.append(b_sample_1)
        b_sample_list_2.append(b_sample_2)

        if len(A_sample_list) == variance_batches_1 and len(A_sample_list) == variance_batches_2:
            # Aggregate samples for intermediate estimate.
            A_sample = np.concatenate(A_sample_list, axis=0).mean(axis=0)
            b_sample_1 = np.concatenate(b_sample_list_1, axis=0).mean(axis=0)
            b_sample_2 = np.concatenate(b_sample_list_2, axis=0).mean(axis=0)
            A_sample_list = []
            b_sample_list_1 = []
            b_sample_list_2 = []

            estimate_list_1.append(calculate_result(A_sample, b_sample_1, total_1))
            estimate_list_2.append(calculate_result(A_sample, b_sample_2, total_2))

            # Estimate current var.
            if len(estimate_list_1) >= min_variance_samples_1:
                var_1 = np.array(estimate_list_1).var(axis=0)

            if len(estimate_list_2) >= min_variance_samples_2:
                var_2 = np.array(estimate_list_2).var(axis=0)

        # Convergence ratio.
        std_1 = np.sqrt(var_1 * variance_batches_1 / (it + 1))
        ratio_1 = np.max(np.max(std_1, axis=0) / (values1.max(axis=0) - values1.min(axis=0)))

        std_2 = np.sqrt(var_2 * variance_batches_2 / (it + 1))
        ratio_2 = np.max(np.max(std_2, axis=0) / (values2.max(axis=0) - values2.min(axis=0)))

        # Print progress message.
        if verbose:
            if detect_convergence:
                print(f'StdDev Ratio = {ratio:.4f} (Converge at {thresh:.4f})')
            else:
                print(f'StdDev Ratio = {ratio:.4f}')

        # Check for convergence.
        if detect_convergence:
            if ratio_1 < thresh and ratio_2 < thresh:
                if verbose:
                    print('Detected convergence')

                # Skip bar ahead.
                if bar:
                    bar.n = bar.total
                    bar.refresh()
                break

        # Forecast number of iterations required.
        if detect_convergence:
            N_est = (it + 1) * (ratio_1 / thresh) ** 2
            if bar and not np.isnan(N_est):
                bar.n = np.around((it + 1) / N_est, 4)
                bar.refresh()
        elif bar:
            bar.update(batch_size)

        
    return ShapleyValues(values1, std_1), ShapleyValues(values2, std_2)