import warnings
import numpy as np
import matplotlib.pyplot as plt


import warnings
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm.auto import tqdm

import warnings
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm.auto import tqdm

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


###########################################################################    FOCUS
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
        if groups is None:
            self.players = num_features
            self.groups_matrix = None
        else:
            # Verify groups.
            inds_list = []
            for group in groups:
                inds_list += list(group)
            assert np.all(np.sort(inds_list) == np.arange(num_features))

            # Map groups to features.
            self.players = len(groups)
            self.groups_matrix = np.zeros((len(groups), num_features), dtype=bool)
            for i, group in enumerate(groups):
                self.groups_matrix[i, group] = True

        # Caching.
        self.sample_repeat = sample

    def __call__(self, S):
        '''
        Evaluate cooperative game.
        Args:
          S: array of player coalitions with size (batch, players).
        '''
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
    '''Calculate the regression coefficients.'''
    num_players = A.shape[1]
    try:
        if len(b.shape) == 2:
            A_inv_one = np.linalg.solve(A, np.ones((num_players, 1)))
        else:
            A_inv_one = np.linalg.solve(A, np.ones(num_players))
        A_inv_vec = np.linalg.solve(A, b)
        values = np.abs((A_inv_vec - A_inv_one * (np.sum(A_inv_vec, axis=0, keepdims=True) - total) / np.sum(A_inv_one))) ########################################################
    except np.linalg.LinAlgError:
        raise ValueError('singular matrix inversion. Consider using larger variance_batches')

    return values

###########################################################################    FOCUS
def ShapleyRegression_U(game,
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
    # Verify arguments.
    # if isinstance(game, CooperativeGame): ###########################################################################    FOCUS
    #     stochastic = False
    # else:
    #     raise ValueError('game must be CooperativeGame')

    if min_variance_samples is None:
        min_variance_samples = default_min_variance_samples(game)
    else:
        assert isinstance(min_variance_samples, int)
        assert min_variance_samples > 1

    if variance_batches is None:
        variance_batches = default_variance_batches(game, batch_size)
    else:
        assert isinstance(variance_batches, int)
        assert variance_batches >= 1

    # Possibly force convergence detection.
    if n_samples is None:
        n_samples = 1e20
        if not detect_convergence:
            detect_convergence = True
            if verbose:
                print('Turning convergence detection on')

    if detect_convergence:
        assert 0 < thresh < 1

    # Weighting kernel (probability of each subset size).
    num_players = game.players
    weights = np.arange(1, num_players)
    weights = 1 / (weights * (num_players - weights))
    weights = weights / np.sum(weights)

    # Calculate null and grand coalitions for constraints.

    null = game.null()
    grand = game.grand()

    # Calculate difference between grand and null coalitions.
    total = grand + null       ##########################################################################

    # Set up bar.
    n_loops = int(np.ceil(n_samples / batch_size))
    if bar:
        if detect_convergence:
            bar = tqdm(total=1)
        else:
            bar = tqdm(total=n_loops * batch_size)

    # Setup.
    n = 0
    b = 0
    A = 0
    estimate_list = []

    # For variance estimation.
    A_sample_list = []
    b_sample_list = []

    # For tracking progress.
    var = np.nan * np.ones(num_players)
    if return_all:
        N_list = []
        std_list = []
        val_list = []

    # Begin sampling.
    for it in range(n_loops):
        # Sample subsets.
        S = np.zeros((batch_size, num_players), dtype=bool)
        num_included = np.random.choice(num_players - 1, size=batch_size, p=weights) + 1
        for row, num in zip(S, num_included):
            inds = np.random.choice(num_players, size=num, replace=False)
            row[inds] = 1


        # Single sample.
        A_sample = np.matmul(S[:, :, np.newaxis].astype(float), S[:, np.newaxis, :].astype(float))

        b_sample = (S.astype(float).T * (game(S) + null)[:, np.newaxis].T).T  ######################################################

        # Welford's algorithm.
        n += batch_size
        b += np.sum(b_sample - b, axis=0) / n    
        A += np.sum(A_sample - A, axis=0) / n

        # Calculate progress.
        values = calculate_result(A, b, total)    
        A_sample_list.append(A_sample)
        b_sample_list.append(b_sample)
        if len(A_sample_list) == variance_batches:
            # Aggregate samples for intermediate estimate.
            A_sample = np.concatenate(A_sample_list, axis=0).mean(axis=0)
            b_sample = np.concatenate(b_sample_list, axis=0).mean(axis=0)
            A_sample_list = []
            b_sample_list = []

            # Add new estimate.
            estimate_list.append(calculate_result(A_sample, b_sample, total))

            # Estimate current var.
            if len(estimate_list) >= min_variance_samples:
                var = np.array(estimate_list).var(axis=0)

        # Convergence ratio.
        std = np.sqrt(var * variance_batches / (it + 1))
        ratio = np.max(
            np.max(std, axis=0) / (values.max(axis=0) - values.min(axis=0)))

        # Print progress message.
        if verbose:
            if detect_convergence:
                print(f'StdDev Ratio = {ratio:.4f} (Converge at {thresh:.4f})')
            else:
                print(f'StdDev Ratio = {ratio:.4f}')

        # Check for convergence.
        if detect_convergence:
            if ratio < thresh:
                if verbose:
                    print('Detected convergence')

                # Skip bar ahead.
                if bar:
                    bar.n = bar.total
                    bar.refresh()
                break

        # Forecast number of iterations required.
        if detect_convergence:
            N_est = (it + 1) * (ratio / thresh) ** 2
            if bar and not np.isnan(N_est):
                bar.n = np.around((it + 1) / N_est, 4)
                bar.refresh()
        elif bar:
            bar.update(batch_size)

        # Save intermediate quantities.
        if return_all:
            val_list.append(values)
            std_list.append(std)
            if detect_convergence:
                N_list.append(N_est)

    # Return results.
    if return_all:
        # Dictionary for progress tracking.
        iters = (
            (np.arange(it + 1) + 1) * batch_size *
            (1 + int(paired_sampling)))
        tracking_dict = {
            'values': val_list,
            'std': std_list,
            'iters': iters}
        if detect_convergence:
            tracking_dict['N_est'] = N_list

        return ShapleyValues(values, std), tracking_dict
    else:
        return ShapleyValues(values, std)