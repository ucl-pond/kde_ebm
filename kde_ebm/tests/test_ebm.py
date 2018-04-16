# Authors: Nicholas C. Firth <ncfirth87@gmail.com>
# License: TBC
from ..Distributions import Gaussian
from .. import MixtureModel
from .. import MCMC
from .. import plotting
import os
import pickle as pkl
from matplotlib import pyplot as plt


def main():
    floc = os.path.dirname(__file__)
    module_loc = os.path.abspath(os.path.join(floc, '../../'))
    X, y = pkl.load(open('%s/data/normal_mixture.pkl' % (module_loc)))
    mixture_models = []
    # for i in range(X.shape[1]):
    #     cn_comp = Gaussian.Gaussian()
    #     ad_comp = Gaussian.Gaussian()
    #     mm = MixtureModel.MixtureModel(cn_comp=cn_comp,
    #                                    ad_comp=ad_comp)
    #     mm.fit(X[:, i], y)
    #     mixture_models.append(mm)
    mixture_models = MixtureModel.fit_all_gmm_models(X, y)
    fig, ax = plotting.mixture_model_grid(X, y, mixture_models)
    fig.show()

    res = MCMC.mcmc(X, mixture_models, n_iter=10)
    fig, ax = plotting.mcmc_uncert_mat(res)
    fig.show()

    sequence = res[0]
    prob_mat = MixtureModel.get_prob_mat(X, mixture_models)
    stages, stages_like = sequence.stage_data(prob_mat)


    plt.show()


if __name__ == '__main__':
    main()
