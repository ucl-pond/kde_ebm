# Authors: Nicholas C. Firth <ncfirth87@gmail.com>
# License: TBC
from ..Distributions import Gaussian
from .. import MixtureModel
import pickle as pkl
import os
from matplotlib import pyplot as plt
import numpy as np


def main():
    floc = os.path.dirname(__file__)
    module_loc = os.path.abspath(os.path.join(floc, '../../'))
    X, y = pkl.load(open('%s/data/normal_mixture.pkl' % (module_loc)))
    mixture_models = []
    for i in range(X.shape[1]):
        cn_comp = Gaussian.Gaussian()
        ad_comp = Gaussian.Gaussian()
        mm = MixtureModel.MixtureModel(cn_comp=cn_comp,
                                       ad_comp=ad_comp)
        mm.fit(X[:, i], y)
        mixture_models.append(mm)
        # fig, ax = plt.subplots()
        # ax.hist([X[y == 0, i], X[y == 1, i]],
        #         alpha=0.6, label=['CN', 'AD'],
        #         normed=True)
        # linspace = np.linspace(X.min(), X.max(), 200)
        # cn_pdf = mm.cn_comp.pdf(linspace)
        # ad_pdf = mm.ad_comp.pdf(linspace)
        # ax.plot(linspace, cn_pdf, color='C0')
        # ax.plot(linspace, ad_pdf, color='C1')
    p_mat = MixtureModel.get_prob_mat(X, y, mixture_models)
    print('p_mat.shape: {0}'.format(p_mat.shape))
    print('p_mat[0, :, :]: {0}'.format(p_mat[0, :, :]))

    # plt.show()


if __name__ == '__main__':
    main()
