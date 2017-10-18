import numpy as np
from ..distributions.gaussian import Gaussian
from ..mixture_model import MixtureModel


def get_prob_mat(X, mixture_models):
    """Gives the matrix of probabilities that a patient has normal/abnormal
    measurements for each of the biomarkers. Output is number of patients x
    number of biomarkers x 2.

    Parameters
    ----------
    X : array-like, shape(numPatients, numBiomarkers)
        All patient-all biomarker measurements.
    y : array-like, shape(numPatients,)
        Diagnosis labels for each of the patients.
    mixtureModels : array-like, shape(numBiomarkers,)
        List of fit mixture models for each of the biomarkers.

    Returns
    -------
    outProbs : array-like, shape(numPatients, numBioMarkers, 2)
        Probability for a normal/abnormal measurement in all biomarkers
        for all patients (and controls).
    """

    prob_mat = np.empty((X.shape[0], X.shape[1], 2))
    for i in range(X.shape[1]):
        prob_mat[:, i, 0] = mixture_models[i].probability(X[:, i])
    prob_mat[:, :, 1] = 1-prob_mat[:, :, 0]
    return prob_mat


def fit_all_gmm_models(X, y):
    n_particp, n_biomarkers = X.shape
    mixture_models = []
    for i in range(n_biomarkers):
        bio_y = y[~np.isnan(X[:, i])]
        bio_X = X[~np.isnan(X[:, i]), i]
        cn_comp = Gaussian()
        ad_comp = Gaussian()
        mm = MixtureModel(cn_comp, ad_comp)
        mm.fit(bio_X, bio_y)
        mixture_models.append(mm)
    return mixture_models