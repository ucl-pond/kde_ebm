# Authors: Neil P. Oxtoby <github:noxtoby>
# License: MIT
import os
import pickle as pkl
import pandas as pd
import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt,cbook as cbook
# plt.rcParams['figure.constrained_layout.use'] = False
plt.rcParams.update({'font.size': 20,
                     'figure.constrained_layout.use': False,
                     "figure.figsize": (30, 30),
                     "axes.labelsize": 'large'})
import seaborn as sns

import warnings
warnings.filterwarnings("ignore",category=cbook.mplDeprecation)

#* KDE-EBM
from kde_ebm.mixture_model import fit_all_kde_models, fit_all_gmm_models, get_prob_mat
from kde_ebm import plotting

def main():
    #floc = os.path.dirname(__file__)
    #module_loc = os.path.abspath(os.path.join(floc, '../../'))
    N_patients = 100
    N_controls = 1000
    N_features = 4
    X_controls, X_patients, X_longitudinal = simulate_some_data(N_patients=N_patients,N_controls=N_controls,N_features=N_features)
    X_p = X_patients #X_patients[X_patients[:,np.floor(N_features/2).astype(int)]<0.5,]
    X1 = np.concatenate( (X_controls, X_p) , axis = 0)
    y1 = np.concatenate( (np.zeros_like(X_controls[:,0]), np.ones_like(X_p[:,0])) , axis = 0).astype(int)
    patholog_dirn_array=np.ones(shape=(1,N_features)).flatten().astype(int).tolist()
    mixture_models_kde = fit_all_kde_models(X1, y1, implement_fixed_controls=True, patholog_dirn_array=patholog_dirn_array)
    #mixture_models_gmm = fit_all_gmm_models(X1, y1)
    figname = 'kde_test-MMgrid-N_p_%i-N_c_%i.png' % (sum(y1==1),sum(y1==0))
    fig, ax = plotting.mixture_model_grid(X1, y1, mixture_models_kde)
    fig.savefig(figname)
    fig.show()
    
    #* Way more controls
    N_controls2 = int(N_controls/2)
    X_controls2 = X_controls[np.random.choice(X_controls.shape[0], N_controls2, replace=False),:]
    X_patients2 = X_patients
    X2 = np.concatenate( (X_controls2, X_patients2) , axis = 0)
    y2 = np.concatenate( (np.zeros_like(X_controls2[:,0]), np.ones_like(X_patients2[:,0])) , axis = 0).astype(int)
    mixture_models_kde2 = fit_all_kde_models(X2, y2, implement_fixed_controls=True, patholog_dirn_array=patholog_dirn_array)
    figname2 = 'kde_test-MMgrid-N_p_%i-N_c_%i.png' % (sum(y2==1),sum(y2==0))
    fig, ax = plotting.mixture_model_grid(X2, y2, mixture_models_kde2)
    fig.savefig(figname2)
    fig.show()
    
    #* score individuals
    msk = (y2==1)
    d = 1
    x = X2[msk,:]
    p_mat_1  = get_prob_mat(x,  mixture_models_kde)
    p_mat_12 = get_prob_mat(x,  mixture_models_kde2)
    # p_mat_2 = get_prob_mat(X2, mixture_models_kde2)
    # p_mat_21 = get_prob_mat(X2, mixture_models_kde)
    fig, (ax,ax2) = plt.subplots(1,2,figsize=(16,6))
    for f in range(N_features):
      ax.plot(np.sort(p_mat_1[:,f,d].flatten()),p_mat_12[:,f,d].flatten()[np.argsort(p_mat_1[:,f,d].flatten())],'.:',markersize=14,label='BM%i' % (f+1))
      ax2.plot(x,p_mat_1[:,f,d].flatten(), '.',markersize=14,label='BM%i matched' % (f+1))
      ax2.plot(x,p_mat_12[:,f,d].flatten(),'x',markersize=14,label='BM%i mismatched' % (f+1))
    ax.legend()
    ax.set_title('P(event)')
    ax.set_xlabel('matched (%i controls, %i patients)' % (N_controls,N_patients))
    ax.set_ylabel('mismatched (%i controls, %i patients)' % (N_controls2,N_patients))
    ax2.set_xlabel('BM value')
    ax2.set_ylabel('p(event)')
    #ax2.legend()
    fig.savefig(figname.replace('MMgrid','p_event')) 
    fig.show()
    
    #print('p_mat.shape: {0}'.format(p_mat_1.shape))
    #print('p_mat[0, :, :]: {0}'.format(p_mat_1[0, :, :]))
    

    plt.show()

def simulate_some_data(N_features = 4, N_patients = 100, N_controls = 100, noise_scale = 0.05, time_scale = 20, N_timepoints_per_individual = 3, plot_bool = False):
    dp = np.linspace(0, time_scale, N_patients)
    dp_long = np.reshape(dp,(-1,1))
    for kt in range(1,N_timepoints_per_individual):
        next_tp = np.reshape(np.linspace(kt, time_scale+kt, N_patients) + np.random.normal(0, 1/24, dp.size),(-1,1))
        dp_long = np.concatenate(
            (dp_long,next_tp),
            axis=1
        )

    def sigmoid(t,a=1,b=-10):
        return 1/(1 + np.exp(-a*(t-b)))

    gradients = np.squeeze(np.broadcast_to(1,(1,N_features)))
    onsets    = np.linspace(0,time_scale,N_features+2)[1:-1]
    
    X_patients = np.empty(shape=(N_patients,N_features))
    X_controls = np.empty(shape=(N_controls,N_features))
    X_longitudinal = np.empty(shape=(N_patients,N_features,N_timepoints_per_individual))
    if plot_bool:
        fig,ax = plt.subplots(figsize=(10,5))
    for a,b,k in zip(gradients,onsets,range(N_features)):
        # print('a = %i, b = %i' % (a,b))
        x = sigmoid(t=dp,a=a,b=b)
        #print(x)
        if plot_bool:
            ax.plot(dp, x)
        #* Longitudinal data
        X_longitudinal[:,k,:] = sigmoid(t=dp_long,a=a,b=b)
        X_longitudinal[:,k,:] += np.random.normal(0, noise_scale, X_longitudinal[:,k,:].shape)
    X_patients = X_longitudinal[:,:,0]
    if plot_bool:
        ax.plot(dp,X_patients,'.')
        ax.set_xlabel("Disease Progression Time",fontsize=20) 
        ax.set_ylabel("sigmoid(t)",fontsize=20)

    #* Sample some controls
    for k in range(len(gradients)):
        X_controls[:,k] = np.random.normal(0, noise_scale, (X_controls.shape[0],))
    
    return X_controls, X_patients, X_longitudinal


if __name__ == '__main__':
    main()
#
#
# def main():
#     floc = os.path.dirname(__file__)
#     module_loc = os.path.abspath(os.path.join(floc, '../../'))
#     X, y = pkl.load(open('%s/data/normal_mixture.pkl' % (module_loc)))
#     mixture_models = []
#     for i in range(X.shape[1]):
#         cn_comp = Gaussian.Gaussian()
#         ad_comp = Gaussian.Gaussian()
#         mm = MixtureModel.MixtureModel(cn_comp=cn_comp,
#                                        ad_comp=ad_comp)
#         mm.fit(X[:, i], y)
#         mixture_models.append(mm)
#         # fig, ax = plt.subplots()
#         # ax.hist([X[y == 0, i], X[y == 1, i]],
#         #         alpha=0.6, label=['CN', 'AD'],
#         #         normed=True)
#         # linspace = np.linspace(X.min(), X.max(), 200)
#         # cn_pdf = mm.cn_comp.pdf(linspace)
#         # ad_pdf = mm.ad_comp.pdf(linspace)
#         # ax.plot(linspace, cn_pdf, color='C0')
#         # ax.plot(linspace, ad_pdf, color='C1')
#     p_mat = MixtureModel.get_prob_mat(X, y, mixture_models)
#     print('p_mat.shape: {0}'.format(p_mat.shape))
#     print('p_mat[0, :, :]: {0}'.format(p_mat[0, :, :]))
#
#     # plt.show()
