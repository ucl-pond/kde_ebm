# Authors: Nicholas C. Firth <ncfirth87@gmail.com>
# License: TBC
from ..event_order import EventOrder
from ..mixture_model import get_prob_mat, fit_all_gmm_models
from multiprocessing import Pool, cpu_count
from itertools import permutations
from ..plotting import mixture_model_grid, mcmc_trace, greedy_ascent_trace
import numpy as np


def greedy_ascent_creation(prob_mat, n_iter=1000, n_init=10):
    n_biomarkers = prob_mat.shape[1]
    starts_dict = dict((x, []) for x in range(n_init))
    for start_idx in range(n_init):
        current_order = EventOrder(n_biomarkers=n_biomarkers)
        current_order.score_ordering(prob_mat)
        starts_dict[start_idx].append(current_order)
        for iter_n in range(1, n_iter):
            new_order = current_order.swap_events()
            new_order.score_ordering(prob_mat)
            if new_order > current_order:
                current_order = new_order
            starts_dict[start_idx].append(current_order)
    return starts_dict


def mcmc(X, mixture_models, n_iter=100000, greedy_n_iter=1000,
         greedy_n_init=10, plot=True):

    prob_mat = get_prob_mat(X, mixture_models)
    greedy_dict = greedy_ascent_creation(prob_mat,
                                         greedy_n_iter,
                                         greedy_n_init)
    if plot:
        fig, ax = greedy_ascent_trace(greedy_dict)
        fig.show()
    current_order = greedy_dict[0][-1]
    for i in range(1, greedy_n_init):
        new_order = greedy_dict[i][-1]
        if new_order > current_order:
            current_order = new_order
    mcmc_samples = [current_order]
    for i in range(1, n_iter):
        new_order = current_order.swap_events()
        new_order.score_ordering(prob_mat)
        ratio = np.exp(new_order - current_order)
        if ratio > np.random.random():
            current_order = new_order
        mcmc_samples.append(current_order)
    mcmc_samples.sort(reverse=True)
    return mcmc_samples


def create_bootstrap(X, y):
    if np.bincount(y).shape > 2:
        raise NotImplementedError(('Only binary labels'
                                   'are currently supported'))
    n_particp, n_biomarkers = X.shape
    boot_X = np.empty(X.shape)
    boot_y = np.empty(y.shape, dtype='int32')
    idxs = np.arange(y.shape[0])

    for i in range(2):
        sample = np.random.choice(idxs[y == i])
        boot_X[i, :] = X[sample, :]
        boot_y[i] = y[sample]
    samples = np.random.choice(idxs, size=y.shape[0]-2)
    boot_X[2:, :] = X[samples, :]
    boot_y[2:] = y[samples]
    iqr = np.nanpercentile(boot_X, 75, axis=0)
    iqr -= np.nanpercentile(boot_X, 25, axis=0)
    if np.any(iqr == 0):
        return create_bootstrap(X, y)
    return boot_X, boot_y


def bootstrap_ebm(X, y, n_bootstrap=50, score_names=None, plot=False):
    bootstrap_samples = []
    for i in range(n_bootstrap):
        boot_X, boot_y = create_bootstrap(X, y)
        kde_mixtures = fit_all_gmm_models(boot_X, boot_y)
        mcmc_samples = mcmc(boot_X, kde_mixtures, plot=False)
        bootstrap_samples += mcmc_samples
        if plot:
            fig, ax = mixture_model_grid(boot_X, boot_y,
                                         kde_mixtures, score_names)
            fig.savefig('Boostrap{}_mixtures.png'.format(i+1))
            fig.close()
            fig, ax = mcmc_trace(mcmc_samples)
            fig.savefig('Boostrap{}_mcmc_trace.png'.format(i+1))
            fig.close()
    return bootstrap_samples


def parallell_bootstrap(X, y, n_bootstrap=50,
                        n_processes=-1):
    bootstrap_samples = []
    for i in range(n_bootstrap):
        bootstrap_samples.append(create_bootstrap(X, y))
    if n_processes == -1:
        n_processes = cpu_count()
    pool = Pool(processes=n_processes)
    mcmc_samples = pool.map(parallell_bootstrap_, bootstrap_samples)
    samples_formatted = []
    for i in range(n_bootstrap):
        samples_formatted += mcmc_samples[0]
        del mcmc_samples[0]
    return samples_formatted


def parallell_bootstrap_(Xy):
    boot_X, boot_y = Xy
    kde_mixtures = fit_all_gmm_models(boot_X, boot_y)
    mcmc_samples = mcmc(boot_X, kde_mixtures, plot=False)
    return mcmc_samples


def enumerate_all(X, mixture_models):
    prob_mat = get_prob_mat(X, mixture_models)
    best_score = -1e10
    best_order = None
    enumerate_samples = []
    for sequence in permutations(np.arange(X.shape[1])):
        event_order = EventOrder(ordering=np.array(sequence))
        sequence_score = event_order.score_ordering(prob_mat)
        enumerate_samples.append(event_order)
        if sequence_score > best_score:
            best_score = sequence_score
            best_order = event_order
    return best_order
