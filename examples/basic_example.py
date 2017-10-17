# Authors: Nicholas C. Firth <ncfirth87@gmail.com>
# License: TBC
from ebm import mixture_model
from ebm import mcmc
from ebm import plotting
from ebm import datasets
import os
import pickle as pkl
from matplotlib import pyplot as plt


def main():
    X, y, cname, bmname = datasets.load_synthetic('synthetic_400_2.csv')
    mixture_models = []
    mixture_models = mixture_model.fit_all_gmm_models(X, y)
    fig, ax = plotting.mixture_model_grid(X, y, mixture_models,
                                          score_names=bmname,
                                          class_names=cname)
    fig.show()

    res = mcmc.mcmc(X, mixture_models, n_iter=10)
    fig, ax = plotting.mcmc_uncert_mat(res, score_names=bmname)
    fig.show()

    sequence = res[0]
    prob_mat = mixture_model.get_prob_mat(X, mixture_models)
    stages, stages_like = sequence.stage_data(prob_mat)

    fig, ax = plotting.stage_histogram(stages, y, )
    plt.show()


if __name__ == '__main__':
    main()
