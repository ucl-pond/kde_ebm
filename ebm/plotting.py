import numpy as np
from matplotlib import pyplot as plt

colors = ['C{}'.format(x) for x in range(10)]


def greedy_ascent_trace(greedy_dict):
    fig, ax = plt.subplots()
    for key, value in greedy_dict.iteritems():
        scores = [x.score for x in value]
        iter_n = np.arange(len(scores))+1
        ax.plot(iter_n, scores, label='Init {}'.format(key+1))
    ax.legend(loc=0)
    fig.suptitle('Greedy Ascent Traces')
    return fig, ax


def mixture_model_grid(X, y, mixtures, score_names=None):
    n_particp, n_biomarkers = X.shape
    if(score_names is None):
        score_names = ['BM{}'.format(x+1) for x in xrange(n_biomarkers)]

    n_x = np.round(np.sqrt(n_biomarkers)).astype(int)
    n_y = np.ceil(np.sqrt(n_biomarkers)).astype(int)
    fig, ax = plt.subplots(n_y, n_x, figsize=(10, 10))
    for i in xrange(n_biomarkers):
        bio_X = X[:, i]
        bio_y = y[~np.isnan(bio_X)]
        bio_X = bio_X[~np.isnan(bio_X)]

        hist_dat = [bio_X[bio_y == 0],
                    bio_X[bio_y == 1]]
        labels = ['CN', 'AD']
        hist_c = colors[:2]
        if(2 in y):
            hist_dat.append(bio_X[bio_y == 2])
            labels.append('MCI')
            hist_c.append(colors[2])
        leg1 = ax[i // n_x, i % n_x].hist(hist_dat,
                                          label=labels,
                                          normed=True,
                                          color=hist_c,
                                          alpha=0.7,
                                          stacked=True)
        linspace = np.linspace(bio_X.min(), bio_X.max(), 100).reshape(-1, 1)
        # controls_score, patholog_score = mixtures[i].pdf(linspace)
        controls_score = mixtures[i].cn_comp.pdf(linspace)
        patholog_score = mixtures[i].ad_comp.pdf(linspace)
        probability = mixtures[i].probability(linspace)
        probability *= np.max((patholog_score, controls_score))
        ax[i // n_x, i % n_x].plot(linspace, controls_score,
                                   color=colors[0])
        ax[i // n_x, i % n_x].plot(linspace, patholog_score,
                                   color=colors[1])
        leg2 = ax[i // n_x, i % n_x].plot(linspace, probability,
                                          color=colors[4])
        ax[i // n_x, i % n_x].set_title(score_names[i])
        ax[i // n_x, i % n_x].axes.get_yaxis().set_visible(False)
    i += 1
    for j in xrange(i, n_x*n_y):
        fig.delaxes(ax[j // n_x, j % n_x])
    fig.legend(leg1[2]+leg2, labels + ['Probability'],
               loc='lower right', fontsize=15)
    fig.tight_layout()
    return fig, ax


def mcmc_trace(mcmc_samples):
    scores = [x.score for x in mcmc_samples]
    iter_n = np.arange(len(scores))+1
    fig, ax = plt.subplots()
    ax.plot(iter_n, scores)
    ax.set_ylabel('Likelihood')
    ax.set_xlabel('Iteration Number')
    fig.suptitle('MCMC Trace')
    return fig, ax


def mcmc_uncert_mat(mcmc_samples, ml_order=None, score_names=None):
    if(ml_order is None):
        ml_order = mcmc_samples[0].ordering
    n_biomarkers = ml_order.shape[0]
    if(score_names is None):
        score_names = ['BM{}'.format(x+1) for x in xrange(n_biomarkers)]
    all_orders = [x.ordering for x in mcmc_samples]
    all_orders = np.array(all_orders)
    confusion_mat = np.zeros((n_biomarkers, n_biomarkers))
    for i in xrange(n_biomarkers):
        confusion_mat[i, :] = np.sum(all_orders == ml_order[i], axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(confusion_mat, interpolation='nearest', cmap='Greys')

    tick_marks = np.arange(n_biomarkers)

    ax.set_xticks(tick_marks)
    ax.set_xticklabels(range(1, n_biomarkers+1), rotation=45)
    trimmed_scores = [x[2:].replace('_', ' ') if x.startswith('p_')
                      else x.replace('_', ' ') for x in score_names]
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(np.array(trimmed_scores, dtype='object')[ml_order],
                       rotation=30, ha='right',
                       rotation_mode='anchor')

    ax.set_ylabel('Biomarker Name', fontsize=20)
    ax.set_xlabel('Event Order', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    fig.tight_layout()

    return fig, ax


def stage_histogram(stages, y, max_stage, str_diag):
    fig, ax = plt.subplots()
    hist_dat = [stages[y == 0],
                stages[y == 1]]
    labels = ['Young Adult', str_diag]
    hist_c = colors[:2]
    if(2 in y):
        hist_dat.append(stages[y == 2])
        labels.append('MCI')
        hist_c.append(colors[2])
    ax.hist(hist_dat,
            label=labels,
            normed=True,
            color=hist_c,
            stacked=False,
            bins=max_stage)
    ax.legend(loc=0, fontsize=20)

    idxs = np.arange(max_stage)
    ax.set_xticks(idxs)
    ax.set_xticklabels([str(x) for x in idxs])

    ax.set_ylabel('Fraction', fontsize=20)
    ax.set_xlabel('EBM Stage', fontsize=20)
    for label in ax.xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=13)
    fig.tight_layout()
    return fig, ax
