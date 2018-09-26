# Authors: Nicholas C. Firth <ncfirth87@gmail.com>
# License: TBC
from ebm import mixture_model
from ebm import mcmc
from ebm import plotting
from ebm import datasets
from ebm import distributions
from matplotlib import pyplot as plt


def main():
    X, y, bmname, cname = datasets.load_synthetic('synthetic_1500_10.csv')
    mixture_models = []
    for i in range(X.shape[1]):
        h_model = distributions.Gaussian()
        d_model = distributions.Gaussian()
        gmm = mixture_model.MixtureModel(cn_comp=h_model,
                                         ad_comp=d_model)
        gmm.fit(X[:, i], y)
        mixture_models.append(gmm)
    fig, ax = plotting.mixture_model_grid(X, y, mixture_models,
                                          score_names=bmname,
                                          class_names=cname)
    fig.show()

    samples = mcmc.mcmc(X, mixture_models, n_iter=200,
                        greedy_n_iter=10, greedy_n_init=2)
    samples.sort(reverse=True)
    ml_order = samples[0]
    fig, ax = plotting.mcmc_uncert_mat(samples, score_names=bmname)
    fig.show()

    bs_samples = mcmc.bootstrap_ebm(X, y, n_mcmc_iter=200,
                                    n_bootstrap=10, greedy_n_init=2,
                                    greedy_n_iter=10)
    fig, ax = plotting.mcmc_uncert_mat(bs_samples, ml_order=ml_order,
                                       score_names=bmname)
    fig.show()
    plt.show()


if __name__ == '__main__':
    import numpy
    numpy.random.seed(42)
    main()
