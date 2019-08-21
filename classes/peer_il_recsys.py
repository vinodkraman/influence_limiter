"""
The primary class used for the Influence Limiter recommender system
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

import sys

from math import isnan

from .recsys_base import *
from .knn import *

#FOCUS HERE, Ifluence Recys srecommender system, meta class, specifiy what black box algoruthm, and then it wilc reate instance of that class, 
#default is KNN, same functions, and plug it right now in
#each item has its only label prior and conditional signal prior as matrices. Dimension is given by number of users
class PeerILRecSys:

    def __init__(self, entry_list, black_box, params, item_conditional_signal_prior, item_target_label_prior, n=1, c=2, loss_type="squared", tracked_user=None):
        '''
        entry_list: list of initial ratings format (user, item, rating, timestamp)
        black_box: type of blackbox to use. Currently only accepts "knn"
        params: parameters for the black box model. Has to be dictionary
        n, c: influence-limiting hyperparameters
        loss_type: type of loss for calculating reputations. Currently only accepts "squared"
        tracked_user: user with respect whom to track reputations/impacts. None defaults to no tracking
        '''

        # Steps for initializing black-box model
        if black_box == "knn":
            self.recsys = KnnClass(entry_list, params)

        # Reputations
        self.reps = {} # reps[truster][trustee]

        # Influence Limiter System
        self.c = c
        self.n = n
        self.loss = loss_type

        #priors
        self.item_target_label_prior = item_target_label_prior #item -> vector of prob where each column corresponds to label
        self.user_item_to_target_label_prior = {} #target_user -> ref_user -> item -> vector of prob where each column corresponds to label
        self.item_conditional_signal_prior = item_conditional_signal_prior #item -> matrix of conditional probabilities
        self.user_item_to_peer_label_posterior = {} #target_user -> ref_user -> item -> vector of posterior probabilities
        #state -> signal -> prob
        
        

        # Pending
        self.pending = {}

        if tracked_user:
            self.tracked_user = tracked_user
            self.tracked_reps = {}
            self.tracked_impact = {}
            self.tracked_counter = 0
            self.reset_tracking()
        else:
            self.tracked_user = None
        


    def reset_tracking(self):
        # Reset tracking stuff (for graphing purposes)
        self.tracked_reps = {i: [self._get_rep(self.tracked_user, i)] for i in self.recsys.user_dict.keys()}
        self.tracked_impact = {i: [0] for i in self.recsys.user_dict.keys()}
        self.tracked_counter = 0


    def _init_rep(self):
        # Initialize reputation based on system's hyperparameters
        return np.exp(-np.log(self.c*self.n))


    def _get_rep(self, target_user, ref_user):
        # Return the reputation of ref_user w.r.t. target_user
        if target_user not in self.reps:
            self.reps[target_user] = {}
        if ref_user not in self.reps[target_user]:
            self.reps[target_user][ref_user] = self._init_rep()
        return self.reps[target_user][ref_user]
    
    def _update_item_target_label_prior(self, t_user, t_item, ref_user, signal):

        target_label_prior = np.array(self.item_target_label_prior[t_item])
        conditional_signal_prior = np.array(self.item_conditional_signal_prior[t_item])

        posterior_matrix = np.divide(np.transpose(conditional_signal_prior * target_label_prior), np.matmul(conditional_signal_prior, target_label_prior))
        label_posterior = []
        if signal == -1:
            label_posterior = posterior_matrix[:,0].tolist()
        else:
            label_posterior = posterior_matrix[:,1].tolist()

        self.item_target_label_prior[t_item] = label_posterior

        if t_user not in self.user_item_to_target_label_prior:
            self.user_item_to_target_label_prior[t_user] = {}
        if ref_user not in self.user_item_to_target_label_prior[t_user]:
            self.user_item_to_target_label_prior[t_user][ref_user] = {}

        self.user_item_to_target_label_prior[t_user][ref_user][t_item] = label_posterior
    
    def _update_user_item_peer_signal_posterior(self, t_user, t_item, ref_user):
        target_label_prior = np.array(self.user_item_to_target_label_prior[t_user][ref_user][t_item])
        conditional_signal_prior = np.array(self.item_conditional_signal_prior[t_item])

        if t_user not in self.user_item_to_peer_label_posterior:
            self.user_item_to_peer_label_posterior[t_user] = {}
        if ref_user not in self.user_item_to_peer_label_posterior[t_user]:
            self.user_item_to_peer_label_posterior[t_user][ref_user] = {}
        
        self.user_item_to_peer_label_posterior[t_user][ref_user][t_item] = np.matmul(conditional_signal_prior, target_label_prior).tolist()

    def recommend(self, t_user, t_item, visualize=False):
        # Process of returning the recommendation of t_item to t_user

        # 1  Do Q and Q_Tilda Procedure
        q_s, q_tmos, q_tildas, betas, user_id_idx, user_ratings = self._make_qs(t_user, t_item)


        if len(q_s) > 0:
            # 2 Add pending
            pend_dict = {
                'user_id_idx': user_id_idx,
                'q_s': q_s,
                'q_tmos': q_tmos,
                'betas': betas,
                'q_tildas': q_tildas, 
                'user_ratings': user_ratings
            }
            self.pending[(t_user, t_item)] = pend_dict

            # 4 Return last q_tilda and q
            last_q_t = q_tildas[-1]
            last_q = q_s[-1]

            # print(f"t_item: {t_item}")
            # print(f"q_tmos: {q_tmos}")
            # print(f"q_s: {q_s}")

            if visualize:
                self._visualize(q_s, q_tmos, betas, q_tildas)
        else:
            last_q_t = self.recsys.prior(t_user, t_item)
            last_q = last_q_t

        # last_q_t: last q tilde (IL recommendation)
        # last_q: last non-IL recommendation
        return last_q_t, last_q


    def _visualize(self, q_s, q_tmos, betas, q_tildas, similarity=[]):
        # Visualization for a single prediction
        n_raters = len(q_s)
        x = np.array(range(1, n_raters+1))

        # attacker_idx = []
        # for idx, user in enumerate(self.user_list):
        #     if user < 0:
        #         attacker_idx.append(idx)

        # for attacker in attacker_idx:
        #     x1 = [attacker+1-0.25, attacker+1+0.25]
        #     y1 = [1,1]
        #     y2 = [-1, -1]
        #     plt.fill_between(x1,y1,y2,interpolate=True, facecolor='red', alpha=0.5)
        plt.plot(x, q_s, color='b', label=r'q_s')
        plt.plot(x, q_tmos, color='g', label=r'q_tmos')
        plt.plot(x, betas, color='k', label=r'\beta_s')
        plt.plot(x, q_tildas, color='y', label=r'\tilde{q}_s')

        if len(similarity) == len(q_s):
            plt.plot(x, similarity, color='r', label=r'similarity')
        plt.scatter(x, q_s, color='b')
        plt.scatter(x, q_tmos, color='g')
        plt.scatter(x, betas, color='k')
        plt.scatter(x, q_tildas, color='y')
        plt.xlabel("users")
        plt.legend()
        plt.show()


    def _make_qs(self, t_user, t_item):
        ''' Iterates and returns q's, q_tmo's, q_t's, beta's'''
        # Initialize
        ratings = self.recsys.set_up(t_user, t_item)
        user_id_idx = {user[0]: idx for idx, user in enumerate(ratings)}

        q_s = []
        q_tmos = []
        betas = []
        q_tildas = []
        q_tmo = self.recsys.prior(t_user, t_item)
        user_ratings = {}

        for index, (user, rating) in enumerate(ratings):
            self.recsys.add_sequential_rating(user, t_item, rating)
            r = self.recsys.make_recommendation(t_user, t_item)
            r = min(r, 1)
            r = max(r, -1)
            q = (r - (-1) ) / (1 - (-1))

            #update posterior label distribution
            self._update_item_target_label_prior(t_user,t_item, user, rating)

            #update posterior over user k's dist
            self._update_user_item_peer_signal_posterior(t_user, t_item, user)

            beta = min(1, self._get_rep(t_user, user))
            q_t = (1 - beta)* q_tmo + beta * q
            q_s.append(q)
            q_tmos.append(q_tmo)
            betas.append(beta)
            q_tildas.append(q_t)
            q_tmo = q_t
            user_ratings[user] = rating
        q_s = np.array(q_s)
        q_tmos = np.array(q_tmos)
        q_tildas = np.array(q_tildas)
        betas = np.array(betas)
        return q_s, q_tmos, q_tildas, betas, user_id_idx, user_ratings


    def receive_label(self, t_user, item, rating):
        '''
        Adds in new rating to dictionaries.
        Updates reputations if applicable
        '''
        if (t_user, item) in self.pending:
            pend_dict = self.pending[(t_user, item)]
            user_ratings = pend_dict['user_ratings']
            raters = list(pend_dict['user_id_idx'].keys()) 
            print(len(raters))
            delta_reps = {}
            for index, rater in enumerate(raters):
                peer_index = (index + 1) % (len(raters)-1)
                print(f"peer_index {peer_index}")
                peer_rater = raters[peer_index]
                print(f"peer_rater {peer_rater}")
                print(f"next_peer_index {peer_index + 1}")
                next_peer_rater = raters[peer_index + 1]

                peer_rating = user_ratings[peer_rater]
                next_peer_rating = user_ratings[next_peer_rater]

                loss_peer = self._losses(self.user_item_to_peer_label_posterior[t_user][rater][item][1], peer_rating)
                loss_next_peer = self._losses(self.user_item_to_peer_label_posterior[t_user][peer_rater][item][1], next_peer_rating)
                
                delta_reps[rater] =  pend_dict['betas'][index] * (loss_peer - loss_next_peer)
            # rep_changes = pend_dict['betas'] * (self._losses(pend_dict['q_tmos'], rating) - self._losses(pend_dict['q_s'], rating))

            for user, user_idx in pend_dict['user_id_idx'].items():
                self.reps[t_user][user] += delta_reps[user]


            last_q_t = pend_dict["q_tildas"][-1]
            last_q = pend_dict["q_s"][-1]

            # Calculate loss
            # Calculate acc
            il_loss = self._losses(last_q_t, rating)
            il_acc = self._accuracy(last_q_t, rating)

            # Calculate NIL loss
            # Calculate NIL acc
            nil_loss = self._losses(last_q, rating)
            nil_acc = self._accuracy(last_q, rating)

            if self.tracked_user:
                if self.tracked_user == t_user:
                    impacts = self._calc_myopic_effect(pend_dict["q_tildas"], pend_dict["q_tmos"], rating)
                else:
                    impacts = np.zeros(len(pend_dict["q_tildas"]))
                for user in self.recsys.user_dict.keys():
                    if user not in self.tracked_reps:
                        self.tracked_reps[user] = [self._init_rep() for i in range(self.tracked_counter + 1)]
                        self.tracked_impact[user] = [0 for i in range(self.tracked_counter + 1)]                        

                    self.tracked_reps[user].append(self._get_rep(self.tracked_user, user))
                    # self.tracked_reps[user].append(self.reps[t_user][user])

                    if user in pend_dict["user_id_idx"]:
                        idx = pend_dict['user_id_idx'][user]
                        self.tracked_impact[user].append(self.tracked_impact[user][-1] + impacts[idx])
                    else:
                        self.tracked_impact[user].append(self.tracked_impact[user][-1])

                self.tracked_counter += 1           

            del self.pending[(t_user, item)]
        else:
            il_loss = -1
            il_acc = -1
            nil_loss = -1
            nil_acc = -1

        self.recsys.receive_rating(t_user, item, rating)

        return il_loss, il_acc, nil_loss, nil_acc


    def return_rep_impacts(self):
        # Returns tracked reputations and tracked impact
        return self.tracked_reps, self.tracked_impact


    def _calc_myopic_effect(self, qtildas, qtmos, rating):
        # calculates myopic effects
        return self._losses(qtmos, rating)-self._losses(qtildas, rating)


    def _losses(self, q_s, label):
        # Calculates losses for a vector of q's
        if self.loss == "squared":
            if label == 1:
                losses = np.square(1 - q_s)
            else:
                losses = np.square(q_s)
        return losses

    def _accuracy(self, q, label):
        # Returns accuracy (0 or 1)
        pred_label = 1 if q >= 0.50 else -1
        return int(label == pred_label)