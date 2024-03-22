import warnings
import numpy as np
import matplotlib.pyplot as plt


def plot(shapley_values,
         feature_names=None,
         sort_features=True,
         max_features=np.inf,
         orientation='horizontal',
         error_bars=True,
         color='tab:green',
         title='Feature Importance',
         title_size=20,
         tick_size=16,
         tick_rotation=None,
         axis_label='',
         label_size=16,
         figsize=(10, 7),
         return_fig=False):
    '''
    Plot Shapley values.
    Args:
      shapley_values: ShapleyValues object.
      feature_names: list of feature names.
      sort_features: whether to sort features by their values.
      max_features: number of features to display.
      orientation: horizontal (default) or vertical.
      error_bars: whether to include standard deviation error bars.
      color: bar chart color.
      title: plot title.
      title_size: font size for title.
      tick_size: font size for feature names and numerical values.
      tick_rotation: tick rotation for feature names (vertical plots only).
      label_size: font size for label.
      figsize: figure size (if fig is None).
      return_fig: whether to return matplotlib figure object.
    '''
    # Default feature names.
    if feature_names is None:
        feature_names = ['Feature {}'.format(i) for i in
                         range(len(shapley_values.values))]

    # Sort features if necessary.
    if len(feature_names) > max_features:
        sort_features = True

    # Perform sorting.
    values = shapley_values.values
    std = shapley_values.std
    if sort_features:
        argsort = np.argsort(values)[::-1]
        values = values[argsort]
        std = std[argsort]
        feature_names = np.array(feature_names)[argsort]

    # Remove extra features if necessary.
    if len(feature_names) > max_features:
        feature_names = (list(feature_names[:max_features])
                         + ['Remaining Features'])
        values = (list(values[:max_features])
                  + [np.sum(values[max_features:])])
        std = (list(std[:max_features])
               + [np.sum(std[max_features:] ** 2) ** 0.5])

    # Warn if too many features.
    if len(feature_names) > 50:
        warnings.warn('Plotting {} features may make figure too crowded, '
                      'consider using max_features'.format(
                        len(feature_names)), Warning)

    # Discard std if necessary.
    if not error_bars:
        std = None

    # Make plot.
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    if orientation == 'horizontal':
        # Bar chart.
        ax.barh(np.arange(len(feature_names))[::-1], values,
                color=color, xerr=std)

        # Feature labels.
        if tick_rotation is not None:
            raise ValueError('rotation not supported for horizontal charts')
        ax.set_yticks(np.arange(len(feature_names))[::-1])
        ax.set_yticklabels(feature_names, fontsize=label_size)

        # Axis labels and ticks.
        ax.set_ylabel('')
        ax.set_xlabel(axis_label, fontsize=label_size)
        ax.tick_params(axis='x', labelsize=tick_size)

    elif orientation == 'vertical':
        # Bar chart.
        ax.bar(np.arange(len(feature_names)), values, color=color,
               yerr=std)

        # Feature labels.
        if tick_rotation is None:
            tick_rotation = 45
        if tick_rotation < 90:
            ha = 'right'
            rotation_mode = 'anchor'
        else:
            ha = 'center'
            rotation_mode = 'default'
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=tick_rotation, ha=ha,
                           rotation_mode=rotation_mode,
                           fontsize=label_size)

        # Axis labels and ticks.
        ax.set_ylabel(axis_label, fontsize=label_size)
        ax.set_xlabel('')
        ax.tick_params(axis='y', labelsize=tick_size)

    else:
        raise ValueError('orientation must be horizontal or vertical')

    # Remove spines.
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_title(title, fontsize=title_size)
    plt.tight_layout()

    if return_fig:
        return fig
    else:
        return


def comparison_plot(comparison_values,
                    comparison_names=None,
                    feature_names=None,
                    sort_features=True,
                    max_features=np.inf,
                    orientation='vertical',
                    error_bars=True,
                    colors=('tab:green', 'tab:blue'),
                    title='Feature Importance Comparison',
                    title_size=20,
                    tick_size=16,
                    tick_rotation=None,
                    axis_label='',
                    label_size=16,
                    legend_loc=None,
                    figsize=(10, 7),
                    return_fig=False):
    '''
    Plot comparison between two different ShapleyValues objects.
    Args:
      comparison_values: tuple of ShapleyValues objects to be compared.
      comparison_names: tuple of names for each ShapleyValues object.
      feature_names: list of feature names.
      sort_features: whether to sort features by their Shapley values.
      max_features: number of features to display.
      orientation: horizontal (default) or vertical.
      error_bars: whether to include standard deviation error bars.
      colors: colors for each set of Shapley values.
      title: plot title.
      title_size: font size for title.
      tick_size: font size for feature names and numerical values.
      tick_rotation: tick rotation for feature names (vertical plots only).
      label_size: font size for label.
      legend_loc: legend location.
      figsize: figure size (if fig is None).
      return_fig: whether to return matplotlib figure object.
    '''
    # Default feature names.
    if feature_names is None:
        feature_names = ['Feature {}'.format(i) for i in
                         range(len(comparison_values[0].values))]

    # Default comparison names.
    num_comps = len(comparison_values)
    if num_comps not in (2, 3, 4, 5):
        raise ValueError('only support comparisons for 2-5 sets of values')
    if comparison_names is None:
        comparison_names = ['Shapley Values {}'.format(i) for i in
                            range(num_comps)]

    # Default colors.
    if colors is None:
        colors = ['tab:green', 'tab:blue', 'tab:purple',
                  'tab:orange', 'tab:pink'][:num_comps]

    # Sort features if necessary.
    if len(feature_names) > max_features:
        sort_features = True

    # Extract values.
    values = [shapley_values.values for shapley_values in comparison_values]
    std = [shapley_values.std for shapley_values in comparison_values]

    # Perform sorting.
    if sort_features:
        argsort = np.argsort(values[0])[::-1]
        values = [shapley_values[argsort] for shapley_values in values]
        std = [stddev[argsort] for stddev in std]
        feature_names = np.array(feature_names)[argsort]

    # Remove extra features if necessary.
    if len(feature_names) > max_features:
        feature_names = (list(feature_names[:max_features])
                         + ['Remaining Features'])
        values = [
            list(shapley_values[:max_features])
            + [np.sum(shapley_values[max_features:])]
            for shapley_values in values]
        std = [list(stddev[:max_features])
               + [np.sum(stddev[max_features:] ** 2) ** 0.5]
               for stddev in std]

    # Warn if too many features.
    if len(feature_names) > 50:
        warnings.warn('Plotting {} features may make figure too crowded, '
                      'consider using max_features'.format(
                        len(feature_names)), Warning)

    # Discard std if necessary.
    if not error_bars:
        std = [None for _ in std]

    # Make plot.
    width = 0.8 / num_comps
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    if orientation == 'horizontal':
        # Bar chart.
        enumeration = enumerate(zip(values, std, comparison_names, colors))
        for i, (shapley_values, stddev, name, color) in enumeration:
            pos = - 0.4 + width / 2 + width * i
            ax.barh(np.arange(len(feature_names))[::-1] - pos,
                    shapley_values, height=width, color=color, xerr=stddev,
                    label=name)

        # Feature labels.
        if tick_rotation is not None:
            raise ValueError('rotation not supported for horizontal charts')
        ax.set_yticks(np.arange(len(feature_names))[::-1])
        ax.set_yticklabels(feature_names, fontsize=label_size)

        # Axis labels and ticks.
        ax.set_ylabel('')
        ax.set_xlabel(axis_label, fontsize=label_size)
        ax.tick_params(axis='x', labelsize=tick_size)

    elif orientation == 'vertical':
        # Bar chart.
        enumeration = enumerate(zip(values, std, comparison_names, colors))
        for i, (shapley_values, stddev, name, color) in enumeration:
            pos = - 0.4 + width / 2 + width * i
            ax.bar(np.arange(len(feature_names)) + pos,
                   shapley_values, width=width, color=color, yerr=stddev,
                   label=name)

        # Feature labels.
        if tick_rotation is None:
            tick_rotation = 45
        if tick_rotation < 90:
            ha = 'right'
            rotation_mode = 'anchor'
        else:
            ha = 'center'
            rotation_mode = 'default'
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=tick_rotation, ha=ha,
                           rotation_mode=rotation_mode,
                           fontsize=label_size)

        # Axis labels and ticks.
        ax.set_ylabel(axis_label, fontsize=label_size)
        ax.set_xlabel('')
        ax.tick_params(axis='y', labelsize=tick_size)

    # Remove spines.
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.legend(loc=legend_loc, fontsize=label_size)
    ax.set_title(title, fontsize=title_size)
    plt.tight_layout()

    if return_fig:
        return fig
    else:
        return
    

import pickle
import numpy as np


def crossentropyloss(pred, target):
    '''Cross entropy loss that does not average across samples.'''
    if pred.ndim == 1:
        pred = pred[:, np.newaxis]
        pred = np.concatenate((1 - pred, pred), axis=1)

    if pred.shape == target.shape:
        # Soft cross entropy loss.
        pred = np.clip(pred, a_min=1e-12, a_max=1-1e-12)
        return - np.sum(np.log(pred) * target, axis=1)
    else:
        # Standard cross entropy loss.
        return - np.log(pred[np.arange(len(pred)), target])


def mseloss(pred, target):
    '''MSE loss that does not average across samples.'''
    if len(pred.shape) == 1:
        pred = pred[:, np.newaxis]
    if len(target.shape) == 1:
        target = target[:, np.newaxis]
    return np.sum((pred - target) ** 2, axis=1)


class ShapleyValues:
    '''For storing and plotting Shapley values.'''
    def __init__(self, values, std):
        self.values = values
        self.std = std

    def plot(self,
             feature_names=None,
             sort_features=True,
             max_features=np.inf,
             orientation='horizontal',
             error_bars=True,
             color='C0',
             title='Feature Importance',
             title_size=20,
             tick_size=16,
             tick_rotation=None,
             axis_label='',
             label_size=16,
             figsize=(10, 7),
             return_fig=False):
        '''
        Plot Shapley values.
        Args:
          feature_names: list of feature names.
          sort_features: whether to sort features by their Shapley values.
          max_features: number of features to display.
          orientation: horizontal (default) or vertical.
          error_bars: whether to include standard deviation error bars.
          color: bar chart color.
          title: plot title.
          title_size: font size for title.
          tick_size: font size for feature names and numerical values.
          tick_rotation: tick rotation for feature names (vertical plots only).
          label_size: font size for label.
          figsize: figure size (if fig is None).
          return_fig: whether to return matplotlib figure object.
        '''
        return plotting.plot(
            self, feature_names, sort_features, max_features, orientation,
            error_bars, color, title, title_size, tick_size, tick_rotation,
            axis_label, label_size, figsize, return_fig)

    def comparison(self,
                   other_values,
                   comparison_names=None,
                   feature_names=None,
                   sort_features=True,
                   max_features=np.inf,
                   orientation='vertical',
                   error_bars=True,
                   colors=None,
                   title='Shapley Value Comparison',
                   title_size=20,
                   tick_size=16,
                   tick_rotation=None,
                   axis_label='',
                   label_size=16,
                   legend_loc=None,
                   figsize=(10, 7),
                   return_fig=False):
        '''
        Plot comparison with another set of Shapley values.
        Args:
          other_values: another Shapley values object.
          comparison_names: tuple of names for each Shapley value object.
          feature_names: list of feature names.
          sort_features: whether to sort features by their Shapley values.
          max_features: number of features to display.
          orientation: horizontal (default) or vertical.
          error_bars: whether to include standard deviation error bars.
          colors: colors for each set of Shapley values.
          title: plot title.
          title_size: font size for title.
          tick_size: font size for feature names and numerical values.
          tick_rotation: tick rotation for feature names (vertical plots only).
          label_size: font size for label.
          legend_loc: legend location.
          figsize: figure size (if fig is None).
          return_fig: whether to return matplotlib figure object.
        '''
        return plotting.comparison_plot(
            (self, other_values), comparison_names, feature_names,
            sort_features, max_features, orientation, error_bars, colors, title,
            title_size, tick_size, tick_rotation, axis_label, label_size,
            legend_loc, figsize, return_fig)

    def save(self, filename):
        '''Save Shapley values object.'''
        if isinstance(filename, str):
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise TypeError('filename must be str')

    def __repr__(self):
        with np.printoptions(precision=2, threshold=12, floatmode='fixed'):
            return 'Shapley Values(\n  (Mean): {}\n  (Std):  {}\n)'.format(
                self.values, self.std)


def load(filename):
    
    '''Load Shapley values object.'''
    with open(filename, 'rb') as f:
        shapley_values = pickle.load(f)
        if isinstance(shapley_values, ShapleyValues):
            return shapley_values
        else:
            raise ValueError('object is not instance of ShapleyValues class')
        
import numpy as np


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
    '''
    Cooperative game for an individual example's prediction.
    Args:
      extension: model extension (see removal.py).
      sample: numpy array representing a single model input.
    '''

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
            self.groups_matrix = np.zeros(
                (len(groups), num_features), dtype=bool)
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


class PredictionLossGame(CooperativeGame):
    '''
    Cooperative game for an individual example's loss value.
    Args:
      extension: model extension (see removal.py).
      sample: numpy array representing a single model input.
      label: the input's true label.
      loss: loss function (see utils.py).
    '''

    def __init__(self, extension, sample, label, loss, groups=None):
        # Add batch dimension to sample.
        if sample.ndim == 1:
            sample = sample[np.newaxis]

        # Add batch dimension to label.
        if np.isscalar(label):
            label = np.array([label])

        # Convert label dtype if necessary.
        if loss is crossentropyloss:
            # Make sure not soft cross entropy.
            if (label.ndim <= 1) or (label.shape[1] == 1):
                # Only convert if float.
                if np.issubdtype(label.dtype, np.floating):
                    label = label.astype(int)

        self.extension = extension
        self.sample = sample
        self.label = label
        self.loss = loss

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
            self.groups_matrix = np.zeros(
                (len(groups), num_features), dtype=bool)
            for i, group in enumerate(groups):
                self.groups_matrix[i, group] = True

        # Caching.
        self.sample_repeat = sample
        self.label_repeat = label

    def __call__(self, S):
        '''
        Evaluate cooperative game.
        Args:
          S: array of player coalitions with size (batch, players).
        '''
        # Try to use caching for repeated data.
        if len(S) != len(self.sample_repeat):
            self.sample_repeat = self.sample.repeat(len(S), 0)
            self.label_repeat = self.label.repeat(len(S), 0)
        input_data = self.sample_repeat
        output_label = self.label_repeat

        # Apply group transformation.
        if self.groups_matrix is not None:
            S = np.matmul(S, self.groups_matrix)

        # Evaluate.
        return - self.loss(self.extension(input_data, S), output_label)
    
import numpy as np


class StochasticCooperativeGame:
    '''Base class for stochastic cooperative games.'''

    def __init__(self):
        raise NotImplementedError

    def __call__(self, S, U):
        raise NotImplementedError

    def sample(self, samples):
        '''Sample exogenous random variable.'''
        inds = np.random.choice(self.N, size=samples)
        return tuple(arr[inds] for arr in self.exogenous)

    def iterate(self, batch_size):
        '''Iterate over values for exogenous random variable.'''
        ind = 0
        while ind < self.N:
            yield tuple(arr[ind:(ind + batch_size)] for arr in self.exogenous)
            ind += batch_size

    def grand(self, batch_size):
        '''Get grand coalition value.'''
        N = 0
        mean_value = 0
        ones = np.ones((batch_size, self.players), dtype=bool)
        for U in self.iterate(batch_size):
            N += len(U[0])

            # Update mean value.
            value = self.__call__(ones[:len(U[0])], U)
            mean_value += np.sum(value - mean_value, axis=0) / N

        return mean_value

    def null(self, batch_size):
        '''Get null coalition value.'''
        N = 0
        mean_value = 0
        zeros = np.zeros((batch_size, self.players), dtype=bool)
        for U in self.iterate(batch_size):
            N += len(U[0])

            # Update mean value.
            value = self.__call__(zeros[:len(U[0])], U)
            mean_value += np.sum(value - mean_value, axis=0) / N

        return mean_value


class DatasetLossGame(StochasticCooperativeGame):
    '''
    Cooperative game representing the model's loss over a dataset.
    Args:
      extension: model extension (see removal.py).
      data: array of model inputs.
      labels: array of corresponding labels.
      loss: loss function (see utils.py).
    '''

    def __init__(self, extension, data, labels, loss, groups=None):
        self.extension = extension
        self.loss = loss
        self.N = len(data)
        assert len(labels) == self.N
        self.exogenous = (data, labels)

        # Store feature groups.
        num_features = data.shape[1]
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
            self.groups_matrix = np.zeros(
                (len(groups), num_features), dtype=bool)
            for i, group in enumerate(groups):
                self.groups_matrix[i, group] = True

    def __call__(self, S, U):
        '''
        Evaluate cooperative game.
        Args:
          S: array of player coalitions with size (batch, players).
          U: tuple of arrays of exogenous random variables, each with size
            (batch, dim).
        '''
        # Unpack exogenous random variable.
        if U is None:
            U = self.sample(len(S))
        x, y = U

        # Possibly convert label datatype.
        if self.loss is crossentropyloss:
            # Make sure not soft cross entropy.
            if (y.ndim == 1) or (y.shape[1] == 1):
                if np.issubdtype(y.dtype, np.floating):
                    y = y.astype(int)

        # Apply group transformation.
        if self.groups_matrix is not None:
            S = np.matmul(S, self.groups_matrix)

        # Evaluate.
        return - self.loss(self.extension(x, S), y)


class DatasetOutputGame(StochasticCooperativeGame):
    '''
    Cooperative game representing the model's loss over a dataset, with respect
    to the full model prediction.
    Args:
      extension: model extension (see removal.py).
      data: array of model inputs.
      loss: loss function (see utils.py).
    '''

    def __init__(self, extension, data, loss, groups=None):
        self.extension = extension
        self.loss = loss
        self.N = len(data)
        self.exogenous = (data,)

        # Store feature groups.
        num_features = data.shape[1]
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
            self.groups_matrix = np.zeros(
                (len(groups), num_features), dtype=bool)
            for i, group in enumerate(groups):
                self.groups_matrix[i, group] = True

    def __call__(self, S, U):
        '''
        Evaluate cooperative game.
        Args:
          S: array of player coalitions with size (batch, players).
          U: tuple of arrays of exogenous random variables, each with size
            (batch, dim).
        '''
        # Unpack exogenous random variable.
        if U is None:
            U = self.sample(len(S))
        x = U[0]

        # Apply group transformation.
        if self.groups_matrix is not None:
            S = np.matmul(S, self.groups_matrix)

        # Evaluate.
        return - self.loss(self.extension(x, S),
                           self.extension(x, np.ones(x.shape, dtype=bool)))
    
import numpy as np
from tqdm.auto import tqdm


def default_min_variance_samples(game):
    '''Determine min_variance_samples.'''
    return 5


def default_variance_batches(game, batch_size):
    '''
    Determine variance_batches.
    This value tries to ensure that enough samples are included to make A
    approximation non-singular.
    '''
    if isinstance(game, CooperativeGame):
        return int(np.ceil(10 * game.players / batch_size))
    else:
        # Require more intermediate samples for stochastic games.
        return int(np.ceil(25 * game.players / batch_size))


def calculate_result(A, b, total):
    '''Calculate the regression coefficients.'''
    num_players = A.shape[1]
    try:
        if len(b.shape) == 2:
            A_inv_one = np.linalg.solve(A, np.ones((num_players, 1)))
        else:
            A_inv_one = np.linalg.solve(A, np.ones(num_players))
        A_inv_vec = np.linalg.solve(A, b)
        values = (
            A_inv_vec -
            A_inv_one * (np.sum(A_inv_vec, axis=0, keepdims=True) - total)
            / np.sum(A_inv_one))
    except np.linalg.LinAlgError:
        raise ValueError('singular matrix inversion. Consider using larger '
                         'variance_batches')

    return values


def ShapleyRegression(game,
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
    if isinstance(game, CooperativeGame):
        stochastic = False
    elif isinstance(game, StochasticCooperativeGame):
        stochastic = True
    else:
        raise ValueError('game must be CooperativeGame or '
                         'StochasticCooperativeGame')

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
    if stochastic:
        null = game.null(batch_size=batch_size)
        grand = game.grand(batch_size=batch_size)
    else:
        null = game.null()
        grand = game.grand()

    # Calculate difference between grand and null coalitions.
    total = grand - null

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
        num_included = np.random.choice(num_players - 1, size=batch_size,
                                        p=weights) + 1
        for row, num in zip(S, num_included):
            inds = np.random.choice(num_players, size=num, replace=False)
            row[inds] = 1

        # Sample exogenous (if applicable).
        if stochastic:
            U = game.sample(batch_size)

        # Update estimators.
        if paired_sampling:
            # Paired samples.
            A_sample = 0.5 * (
                np.matmul(S[:, :, np.newaxis].astype(float),
                          S[:, np.newaxis, :].astype(float))
                + np.matmul(np.logical_not(S)[:, :, np.newaxis].astype(float),
                            np.logical_not(S)[:, np.newaxis, :].astype(float)))
            if stochastic:
                game_eval = game(S, U) - null
                S_comp = np.logical_not(S)
                comp_eval = game(S_comp, U) - null
                b_sample = 0.5 * (
                    S.astype(float).T * game_eval[:, np.newaxis].T
                    + S_comp.astype(float).T * comp_eval[:, np.newaxis].T).T
            else:
                game_eval = game(S) - null
                S_comp = np.logical_not(S)
                comp_eval = game(S_comp) - null
                b_sample = 0.5 * (
                    S.astype(float).T * game_eval[:, np.newaxis].T +
                    S_comp.astype(float).T * comp_eval[:, np.newaxis].T).T
        else:
            # Single sample.
            A_sample = np.matmul(S[:, :, np.newaxis].astype(float),
                                 S[:, np.newaxis, :].astype(float))
            if stochastic:
                b_sample = (S.astype(float).T
                            * (game(S, U) - null)[:, np.newaxis].T).T
            else:
                b_sample = (S.astype(float).T
                            * (game(S) - null)[:, np.newaxis].T).T

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