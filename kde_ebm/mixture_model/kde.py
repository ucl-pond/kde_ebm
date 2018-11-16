import numpy as np
from sklearn import neighbors


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

    def fit(self, X, y):
        sorted_idx = X.argsort(axis=0).flatten()
        kde_values = X.copy()[sorted_idx].reshape(-1,1)
        kde_labels = y.copy()[sorted_idx]

        bin_counts = np.bincount(y).astype(float)
        mixture = 0.5
        old_ratios = np.zeros(kde_labels.shape)
        iter_count = 0
        if(self.bandwidth is None):
            self.bandwidth = hscott(X)
        for i in range(self.n_iters):
            controls_kde = neighbors.KernelDensity(kernel=self.kernel,
                                                   bandwidth=self.bandwidth)
            patholog_kde = neighbors.KernelDensity(kernel=self.kernel,
                                                   bandwidth=self.bandwidth)
            controls_kde.fit(kde_values[kde_labels == 0])
            patholog_kde.fit(kde_values[kde_labels == 1])

            controls_score = controls_kde.score_samples(kde_values)
            patholog_score = patholog_kde.score_samples(kde_values)

            #* Missing data
            controls_score[np.isnan(controls_score)] = 0.5
            patholog_score[np.isnan(patholog_score)] = 0.5

            controls_score = np.exp(controls_score)*mixture
            patholog_score = np.exp(patholog_score)*(1-mixture)

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
        controls_score = self.controls_kde.score_samples(X)
        controls_score = np.exp(controls_score)*self.mixture
        patholog_score = self.patholog_kde.score_samples(X)
        patholog_score = np.exp(patholog_score)*(1-self.mixture)
        return controls_score, patholog_score

    def probability(self, X):
        controls_score, patholog_score = self.pdf(X.reshape(-1, 1))
        #* Handle missing data
        controls_score[np.isnan(controls_score)] = 0.5
        patholog_score[np.isnan(patholog_score)] = 0.5
        c = controls_score / (controls_score+patholog_score)
        if (c==0) or np.isnan(c):
            return 0.5
        else:
            return c

    def BIC(self, X):
        controls_score, patholog_score = self.pdf(X.reshape(-1, 1))
        likelihood = controls_score + patholog_score
        likelihood = -1*np.log(likelihood).sum()
        return 2*likelihood+2*np.log(X.shape[0])


def hscott(x, weights=None):

    IQR = np.percentile(x, 75) - np.percentile(x, 25)
    A = min(np.std(x, ddof=1), IQR / 1.349)

    if weights is None:
        weights = np.ones(len(x))
    n = float(sum(weights))

    return 1.059 * A * n ** (-0.2)
