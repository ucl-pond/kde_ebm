# Authors: Nicholas C. Firth <ncfirth87@gmail.com>
# License: TBC
from ..Distributions import Gaussian
import pickle as pkl
import os
from scipy import stats
import numpy as np

def main():
    # 
    floc = os.path.dirname(__file__)
    module_loc = os.path.abspath(os.path.join(floc, '../../'))
    X, y = pkl.load(open('%s/data/normal_mixture.pkl' % (module_loc)))
    # for i in range(X.shape):
    # print('{0}, {1}'.format(X.shape, y.shape))
    g = Gaussian.Gaussian(mu=0, sigma=1)
    # print(g.pdf([0, 0.1, 0.5, 1])[0] == 0.39894228)
    event_sign = np.nanmean(X[y == 0, 0]) > np.nanmean(X[y == 1, 0])
    print(np.nanmin(X[y == 0]))
    print(g.get_bounds(X[:, 0], X[y == 0], event_sign))


if __name__ == '__main__':
    main()
