# Authors: Nicholas C. Firth <ncfirth87@gmail.com>
# License: TBC
from ebm import mixture_model
from ebm import mcmc
from ebm import plotting
from ebm import datasets
from matplotlib import pyplot as plt


def main():
    X, y, cname, bmname = datasets.load_synthetic('synthetic_1500_10.csv')
    mixture_models = []
    mixture_models = mixture_model.fit_all_gmm_models(X, y)
    fig, ax = plotting.mixture_model_grid(X, y, mixture_models,
                                          score_names=bmname,
                                          class_names=cname)
    fig.show()

    res = mcmc.mcmc(X, mixture_models, n_iter=500,
                    greedy_n_iter=10, greedy_n_init=2)
    fig, ax = plotting.mcmc_uncert_mat(res, score_names=bmname)
    fig.show()
    ml_order = res[0]
    prob_mat = mixture_model.get_prob_mat(X, mixture_models)
    stages, stages_like = ml_order.stage_data(prob_mat)

    fig, ax = plotting.stage_histogram(stages, y, )
    plt.show()


if __name__ == '__main__':
    import numpy
    numpy.random.seed(42)
    main()
