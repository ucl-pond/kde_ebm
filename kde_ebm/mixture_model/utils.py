import numpy as np
from ..distributions.gaussian import Gaussian
from .gmm import ParametricMM
from .kde import KDEMM


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

    n_particp, n_biomarkers = X.shape
    prob_mat = np.zeros((n_particp, n_biomarkers, 2))
    for i in range(n_biomarkers):
        # Impute missing data s.t. sequence inference is not biassed, i.e. p(x|E) == p(x|!E) and both p!=0
        X_imputed = mixture_models[i].impute_missing(X[:,i].reshape(-1,1))
        probs = mixture_models[i].pdfs_mixture_components(X_imputed)
        prob_mat[:, i, 0] = probs[0].flatten()
        prob_mat[:, i, 1] = probs[1].flatten()
    return prob_mat


def fit_all_gmm_models(X, y, fit_all_subjects=False, implement_fixed_controls=False, patholog_dirn=None, outlier_controls_quantile = 0.9):
    if not fit_all_subjects:
        #* Extract only the first two diagnoses (controls & patients)
        msk = np.where(y<2)[0]
        X = X[msk]
        y = y[msk]
    
    n_particp, n_biomarkers = X.shape
    mixture_models = []
    for i in range(n_biomarkers):
        bio_y = y[~np.isnan(X[:, i])]
        bio_X = X[~np.isnan(X[:, i]), i]
        cn_comp = Gaussian()
        ad_comp = Gaussian()
        mm = ParametricMM(cn_comp, ad_comp)
        if implement_fixed_controls:
            mm.fit_constrained(bio_X,bio_y)
        else:
            mm.fit(bio_X, bio_y)
        mixture_models.append(mm)
    return mixture_models


def fit_all_kde_models(X, y, implement_fixed_controls=False, patholog_dirn_array=None, outlier_controls_quantile = 0.9):
    #* Extract only the first two diagnoses
    msk = np.where(y<2)[0]
    X = X[msk]
    y = y[msk]
    
    n_particp, n_biomarkers = X.shape
    kde_mixtures = []
    for i in range(n_biomarkers):
        patholog_dirn = patholog_dirn_array[i] if patholog_dirn_array is not None else None
        bio_X = X[:, i]
        bio_y = y[~np.isnan(bio_X)]
        bio_X = bio_X[~np.isnan(bio_X)]
        # print('utils:fit_all_kde_models() \n  - range(np.isnan(bio_X[y=0/1])) = [{0},{1}],[{2},{3}]'.format(
        #     min((bio_X[bio_y==0])),max((bio_X[bio_y==0])),
        #     min((bio_X[bio_y==1])),max((bio_X[bio_y==1]))
        #     )
        # )
        kde = KDEMM()
        kde.fit(bio_X, bio_y,implement_fixed_controls=implement_fixed_controls, patholog_dirn=patholog_dirn,outlier_controls_quantile=outlier_controls_quantile)
        kde_mixtures.append(kde)
    return kde_mixtures
