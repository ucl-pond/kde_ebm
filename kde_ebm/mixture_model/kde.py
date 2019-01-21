import numpy as np
from sklearn import neighbors
from awkde import GaussianKDE # from scipy import stats

class KDEMM(object):
    """docstring for KDEMM"""
    def __init__(self, kernel='gaussian', bandwidth=None, n_iters=1500):
        super(KDEMM, self).__init__()
        self.n_iters = n_iters
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.controls_kde = None
        self.patholog_kde = None
        self.mixture = None
        self.alpha = 0.5 # sensitivity parameter: 0...1

    def fit(self, X, y):
        sorted_idx = X.argsort(axis=0).flatten()
        kde_values = X.copy()[sorted_idx].reshape(-1,1)
        kde_labels = y.copy()[sorted_idx]

        bin_counts = np.bincount(y).astype(float)
        mixture = 0.5
        old_ratios = np.zeros(kde_labels.shape)
        iter_count = 0
        if(self.bandwidth is None):
            #* 1. Rule of thumb
            self.bandwidth = hscott(X)
            # #* 2. Estimate full density to inform variable bandwidth: wide in tails, narrow in peaks
            # all_kde = neighbors.KernelDensity(kernel=self.kernel,
            #                                   bandwidth=self.bandwidth)
            # all_kde.fit(kde_values)
            # f = np.exp(all_kde.score_samples(kde_values))
            # #* 3. Local, a.k.a. variable, bandwidth given by eq. 3 of https://ieeexplore.ieee.org/abstract/document/7761150
            # g = stats.mstats.gmean(f)
            # alpha = 0.5 # sensitivity parameter: 0...1
            # lamb = np.power(f/g,-alpha)
        for i in range(self.n_iters):
            # #* Separate bandwidth for each mixture component, recalculated each loop
            # bw_controls = self.bandwidth # hscott(kde_values[kde_labels == 0])
            # bw_patholog = self.bandwidth # hscott(kde_values[kde_labels == 1])
            # controls_kde = neighbors.KernelDensity(kernel=self.kernel,
            #                                        bandwidth=bw_controls)
            # patholog_kde = neighbors.KernelDensity(kernel=self.kernel,
            #                                        bandwidth=bw_patholog)
            # controls_kde.fit(kde_values[kde_labels == 0])
            # # patholog_kde.fit(kde_values[kde_labels == 1])
            # controls_score = controls_kde.score_samples(kde_values)
            # patholog_score = patholog_kde.score_samples(kde_values)
            # #* Missing data - 50/50 likelihood
            # controls_score[np.isnan(controls_score)] = np.log(0.5)
            # patholog_score[np.isnan(patholog_score)] = np.log(0.5)

            #* Automatic variable/local bandwidth
            controls_kde = GaussianKDE(glob_bw="scott", alpha=self.alpha, diag_cov=False)
            patholog_kde = GaussianKDE(glob_bw="scott", alpha=self.alpha, diag_cov=False)
            controls_kde.fit(kde_values[kde_labels == 0])
            patholog_kde.fit(kde_values[kde_labels == 1])

            controls_score = controls_kde.predict(kde_values)
            patholog_score = patholog_kde.predict(kde_values)
            #* Missing data - need to test this
            controls_score[np.isnan(controls_score)] = 0.5
            patholog_score[np.isnan(patholog_score)] = 0.5
            controls_score = controls_score*mixture
            patholog_score = patholog_score*(1-mixture)

            ratio = controls_score / (controls_score + patholog_score)
            if(np.all(ratio == old_ratios)):
                break
            iter_count += 1
            old_ratios = ratio
            kde_labels = ratio < 0.5

            diff_y = np.hstack(([0], np.diff(kde_labels)))
            if (np.sum(diff_y != 0) == 2 and
                    np.unique(kde_labels).shape[0] == 2):
                split_y = int(np.all(np.diff(np.where(kde_labels == 0)) == 1))
                sizes = [x.shape[0] for x in
                         np.split(diff_y, np.where(diff_y != 0)[0])]
                split_prior_smaller = (np.mean(kde_values[kde_labels ==
                                                          split_y])
                                       < np.mean(kde_values[kde_labels ==
                                                            (split_y+1) % 2]))
                if split_prior_smaller:
                    replace_idxs = np.arange(kde_values.shape[0])[-sizes[2]:]
                else:
                    replace_idxs = np.arange(kde_values.shape[0])[:sizes[0]]

                kde_labels[replace_idxs] = (split_y+1) % 2

            bin_counts = np.bincount(kde_labels).astype(float)
            mixture = bin_counts[0] / bin_counts.sum()
            if(mixture < 0.10 or mixture > 0.90):
                break
        self.controls_kde = controls_kde
        self.patholog_kde = patholog_kde
        self.mixture = mixture
        self.iter_ = iter_count
        return self

    def likelihood(self, X):
        controls_score, patholog_score = self.pdf(X)
        data_likelihood = controls_score+patholog_score
        data_likelihood = np.log(data_likelihood)
        return -1*np.sum(data_likelihood)

    def pdf(self, X, **kwargs):
        #* Old version: sklearn fixed-bw KDE
        # controls_score = self.controls_kde.score_samples(X)
        # controls_score = np.exp(controls_score)*self.mixture
        # patholog_score = self.patholog_kde.score_samples(X)
        # patholog_score = np.exp(patholog_score)*(1-self.mixture)
        #* Auto-Variable-bw KDE: awkde
        controls_score = self.controls_kde.predict(X)*self.mixture
        patholog_score = self.patholog_kde.predict(X)*(1-self.mixture)
        return controls_score, patholog_score

    def probability(self, X):
        controls_score, patholog_score = self.pdf(X.reshape(-1, 1))
        #* Handle missing data
        controls_score[np.isnan(controls_score)] = 0.5
        patholog_score[np.isnan(patholog_score)] = 0.5
        controls_score[controls_score==0] = 0.5
        patholog_score[patholog_score==0] = 0.5
        c = controls_score / (controls_score+patholog_score)
        #c[(controls_score+patholog_score)==0] = 0.5
        return c

    def BIC(self, X):
        controls_score, patholog_score = self.pdf(X.reshape(-1, 1))
        likelihood = controls_score + patholog_score
        likelihood = -1*np.log(likelihood).sum()
        return 2*likelihood+2*np.log(X.shape[0])


def hscott(x, weights=None):

    IQR = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
    A = min(np.nanstd(x, ddof=1), IQR / 1.349)

    if weights is None:
        weights = np.ones(len(x))
    n = float(np.nansum(weights))
    #n = n/sum(~np.isnan(x))

    return 1.059 * A * n ** (-0.2)
