# Authors: Nicholas C. Firth <ncfirth87@gmail.com>
# License: TBC
import numpy as np


class EventOrder(object):
    def __init__(self, ordering=None, n_biomarkers=None, score=None):
        super(EventOrder, self).__init__()
        if ordering is None and n_biomarkers is None:
            raise ValueError('EventOrder __init__ takes one argument,'
                             ' zero given')
        if ordering is None:
            self.ordering = np.arange(n_biomarkers)
            np.random.shuffle(self.ordering)
            self.n_biomarkers = n_biomarkers
        else:
            self.ordering = ordering
            self.n_biomarkers = ordering.shape[0]
        self.score = score

    def score_ordering(self, prob_mat):
        likelihoods = self.calc_indiv_likelihoods(prob_mat)
        likelihood = np.sum(likelihoods)
        self.score = likelihood
        return likelihood

    def calc_indiv_likelihoods(self, prob_mat):
        k = prob_mat.shape[1]+1
        p_perm = self.calc_perm_matrix(prob_mat)
        likelihoods = np.log(np.sum((1./k)*p_perm, 1)+1e-250)
        return likelihoods

    def calc_perm_matrix(self, prob_mat):
        event_order = self.ordering
        p_yes = np.array(prob_mat[:, event_order, 1])
        p_no = np.array(prob_mat[:, event_order, 0])

        k = prob_mat.shape[1]+1
        p_perm = np.zeros((prob_mat.shape[0], k))

        for i in range(k):
            p_perm[:, i] = np.prod(p_yes[:, :i], 1)*np.prod(p_no[:, i:k-1], 1)
        return p_perm

    def stage_data(self, prob_mat):
        event_order = self.ordering
        p_yes = np.array(prob_mat[:, event_order, 1])
        p_no = np.array(prob_mat[:, event_order, 0])
        n_particp, n_biomarkers = p_yes.shape
        k = n_biomarkers+1

        stage_likelihoods = np.empty((n_particp, n_biomarkers+1))
        for i in range(k):
            stage_likelihoods[:, i] = np.prod(p_yes[:, :i], 1)*np.prod(p_no[:, i:n_biomarkers], 1)
        # Maximum-Likelihood stage
        stages = np.argmax(stage_likelihoods, axis=1)
        # Weighted-average stage: weighted by stage likelihood
        #stage_likelihoods_max = np.tile(np.max(stage_likelihoods,axis=1).reshape(-1,1),(1,stage_likelihoods.shape[1]))
        #weights = stage_likelihoods / stage_likelihoods_max
        #stages_weighted_avg = np.average(np.tile(np.arange(0,n_biomarkers+1),(n_particp,1)),
        #                                 axis=1,
        #                                 weights=weights)
        return stages, stage_likelihoods

    def swap_events(self):
        event_order = self.ordering
        new_event_order = event_order.copy()
        swap_bm = np.random.choice(event_order.shape[0], 2, replace=False)
        new_event_order[swap_bm] = new_event_order[swap_bm[::-1]]
        return EventOrder(ordering=new_event_order)

    def __eq__(self, other):
        return np.all(self.ordering == other.ordering)

    def __hash__(self):
        return hash(('ordering', self.ordering.tostring()))

    def __lt__(self, other):
        if self.score is None and other.score is None:
            raise ValueError('Cannot compare unscored orderings')
        if self.score < other.score:
            return True
        return False

    def __gt__(self, other):
        if self.score is None and other.score is None:
            raise ValueError('Cannot compare unscored orderings')
        if self.score > other.score:
            return True
        return False

    def __add__(self, other):
        if self.score is None and other.score is None:
            raise ValueError('Cannot subtract unscored orderings')
        return self.score + other.score

    def __sub__(self, other):
        if self.score is None and other.score is None:
            raise ValueError('Cannot subtract unscored orderings')
        return self.score - other.score

    def __repr__(self):
        return 'EventOrder(order=%r, score=%r)' % (self.ordering,
                                                   self.score)

    def __str__(self):
        return self.__repr__()
