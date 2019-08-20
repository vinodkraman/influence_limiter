# A black box algorithm for recommendation using k nearest neighbors

import numpy as np

from .recsys_base import *


class KnnClass(RecSysBase):

    def __init__(self, entry_list, params={"corr": "pearson", "strat": "top_k", "param": 5}):
        # Steps for initializing data

        # Online updating
        self.user_mean = {}
        self.user_std = {}
        self.rating_square_sum = {}
        self.online_calculated_dict = {} # User (m, n) to access, where m is 
                                         # smaller one, and n is larger one

        super().__init__(entry_list)

        # Making prediction
        self.features = []
        self.similarity = []
        self.rater_std = []
        self.user_list = []
        self.t_user = None

        # Model Hyperparameters
        self.corr_type = params["corr"]
        self.strat = params["strat"]
        self.param = params["param"]

        self.epsilon = 0.00001

        self._init_mean_std()
        self._init_online_dict()


    def set_up(self, user, item):
        # sets up for recommending item to user
        super().set_up(user, item)
        self.similarity = []
        self.rater_std = []
        self.features = []
        user_list = self._get_rated_user_list(item)
        self.t_user = user

        if user not in self.user_dict.keys():
            self.add_new_user(user)
            # print("added new user")

        return [(curr_user, self.item_dict[item][curr_user]) for curr_user in user_list]


    def add_new_user(self, user):
        # adds new user to system
        super().add_new_user(user)
        self._init_user_mean_std(user)
        for user2 in self.user_dict.keys():
            if user2 != user:
                self._init_user_target_online_dict(user2, user)
                m = min(user, user2)
                n = max(user, user2)
        #         print(self.online_calculated_dict[(m,n)])
        # print(self.user_mean[user])



    def add_ratings(self, entry_list):
        # Adds a list of entries to the recsys
        super().add_ratings(entry_list)


    def add_sequential_rating(self, rater, item, rating):
        # Given (user, item, rating), incorporates it and updates any
        # necessary data structures

        # Update info
        raterDict = self.user_dict[rater]
        self.rater_std.append(self.user_std[rater])
        self.features.append(raterDict[item] - self.user_mean[rater])

        # Calculate similarity
        similarity = self._calculate_similarity(self.t_user, rater)
        self.similarity.append(similarity)


    def prior(self, user, item):
        # Give prior prediction
        return 0.0


    def make_recommendation(self, user, item):
        # Main function that is used to make recommendations

        super().make_recommendation(user, item)

        if user not in self.user_dict.keys():
            return self.prior(user, item)

        mask = np.array(range(len(self.similarity)))

        if self.strat == "top_k":
            k = min(len(self.similarity), self.param)
            mask = np.argsort(self.similarity)[-k:]
        elif self.strat == "threshold":
            mask = np.abs(np.array(self.similarity)) >= self.param
        elif self.strat == "percentage":
            k = top_percentage*len(self.similarity)
            mask = np.argsort(self.similarity)[-k:]
            
        weighted_feature = np.array(self.similarity)[mask] * np.array(self.features)[mask]
        if np.sum(abs(np.array(self.similarity)[mask])) == 0:
            return self.prior(user, item)

        pred = self.user_mean[user] + \
            self.user_std[user] * \
            np.sum(weighted_feature/ np.array(self.rater_std)[mask]) / np.sum(abs(np.array(self.similarity)[mask]))

        if np.isnan(pred):
            return self.user_mean[user] + \
                self.user_std[user] * \
                np.sum(weighted_feature/ (np.array(self.rater_std) + 0.00001 )[mask]) / np.sum(abs(np.array(self.similarity)[mask]))

        return pred


    def receive_rating(self, user, item, rating, timestamp=None):
        '''
        Adds in new rating to dictionaries.
        Updates reputations if applicable
        '''

        super().receive_rating(user, item, rating, timestamp)

        # Update online stuff

        if user in self.user_mean:
            self._update_user_mean_std(user, item, rating)
        else:
            self._init_user_mean_std(user)

        for user2 in self.item_dict[item].keys():
            if user != user2:
                m = min(user, user2)
                n = max(user, user2)
                if m != n:
                    if (m, n) in self.online_calculated_dict:
                        self._update_online_dict(user, user2, item)
                    else:
                        self._init_user_target_online_dict(user, user2)


    def _calculate_similarity(self, target_user, rater):
        '''
        Calculate the similarity based on the dictionary for target user 
        and target item.
        '''
        if target_user == rater:
            return 1.0
        m = min(target_user, rater)
        n = max(target_user, rater)    
        if (m, n) not in self.online_calculated_dict:
            self._init_user_target_online_dict(target_user, rater)  

        if self.corr_type == "pearson":
            similarity = self._pearson_similarity(target_user, rater)
        else:
            similarity = self._jaccard_similarity(target_user, rater)
        return similarity


    def _update_user_mean_std(self, user, item, rating):
        ru = self.user_mean[user]
        n = self.rating_per_user[user]
        self.user_mean[user] += (1/n)*(rating-ru)
        ru = self.user_mean[user]
        self.rating_square_sum[user] += rating ** 2
        if n - 1 > 0:
            self.user_std[user] = 1/(n-1)*(self.rating_square_sum[user]-n*ru**2)
            self.user_std[user] = np.sqrt(self.user_std[user])


    def _update_online_dict(self, user1, user2, item):
        if user1 == user2:
            return
        m = min(user1, user2)
        n = max(user1, user2)
        online_dict = self.online_calculated_dict[(m, n)]
        rm = self.user_dict[m][item]
        rn = self.user_dict[n][item]
        online_dict["product"] += rm * rn
        online_dict["cardn"] += 1
        online_dict["sum_rm"] += rm
        online_dict["sum_rn"] += rn
        online_dict["sum_rm_square"] += rm ** 2
        online_dict["sum_rn_square"] += rn ** 2
        self.online_calculated_dict[(m,n)] = online_dict


    def _get_rated_user_list(self, item):
        user_list = []
        if item in self.item_dict:
            for user in self.item_dict[item].keys():
                user_list.append((user, self.time_dict[item][user]))
            user_list = sorted(user_list, key=lambda x:x[1])
            user_list = [i[0] for i in user_list]
        return user_list


    def _init_mean_std(self):
        '''
        Called after data is held out
        '''
        self.user_mean = {}
        self.user_std = {}
        for user in self.user_dict.keys():
            self._init_user_mean_std(user)


    def _init_user_mean_std(self, user):
        temp = list(self.user_dict[user].values())
        if len(temp) == 0:
            self.user_mean[user] = 0
            self.rating_square_sum[user] = 0
        else:
            self.user_mean[user] = np.mean(temp)
            self.rating_square_sum[user] = np.sum(np.power(temp, 2))
        if len(temp) > 1:
            self.user_std[user] = np.std(temp, ddof=1)
        else:
            self.user_std[user] = 1
        


    def _init_online_dict(self):
        for user_1 in self.user_dict.keys():
            for user_2 in self.user_dict.keys():
                if user_1 == user_2:
                    continue
                self._init_user_target_online_dict(user_1, user_2)


    def _init_user_target_online_dict(self, target, user):
        if target == user:
            print("Error: target == user!")
            exit(-1)
        m = min(target, user)
        n = max(target, user)
        mdict = self.user_dict[m]
        X = set(mdict.keys())
        ndict = self.user_dict[n]
        Y = set(ndict.keys())
        N = X.intersection(Y)
        self.online_calculated_dict[(m,n)] = {}
        online_dict = {}
        temp_m = []
        temp_n = []
        for i in N:
            temp_m.append(mdict[i])
            temp_n.append(ndict[i])
        temp_m = np.array(temp_m)
        temp_n = np.array(temp_n)
        online_dict["product"] = np.sum(temp_m*temp_n)
        online_dict["cardn"] = len(N)
        online_dict["sum_rm"] = np.sum(temp_m)
        online_dict["sum_rn"] = np.sum(temp_n)
        online_dict["sum_rm_square"] = np.sum(np.power(temp_m, 2))
        online_dict["sum_rn_square"] = np.sum(np.power(temp_n, 2))
        self.online_calculated_dict[(m,n)] = online_dict

    # 2. Similarity
    # Assumption: the dictionary and the usermean/std has been fully updated
    def _jaccard_similarity(self, target_user, rater):
        '''
        Calculate Jaccard similarity.
        In order to make this not problematic, we should ensure the dictionary
        have all 'unseen' items removed.
        '''
        # The item target user has rated
        X = set(self.user_dict[target_user].keys())
        # The item rater has rated
        Y = set(self.user_dict[rater].keys())
        if len(X.union(Y)) == 0:
            return 0
        return len(X.intersection(Y))/len(X.union(Y))


    def _pearson_similarity(self, target_user, rater):
        '''
        Calculate Pearson correlation.
        In order to make this not problematic, we should ensure the dictionary
        have all 'unseen' items removed.
        '''
        m = min(target_user, rater)
        n = max(target_user, rater)
        online_dict = self.online_calculated_dict[(m,n)]
        rmrn = online_dict["product"]
        cardn = online_dict["cardn"]
        rm = online_dict["sum_rm"]
        rn = online_dict["sum_rn"]
        avg_m = self.user_mean[m]
        avg_n = self.user_mean[n]
        rm2 = online_dict["sum_rm_square"]
        rn2 = online_dict["sum_rn_square"]
        if cardn == 0:
            return 0
        if rm2 + cardn*avg_m**2 - 2*avg_m*rm == 0 or rn2 + cardn*avg_n**2 - 2*avg_n*rn == 0:
            return 0
        return (rmrn + cardn * avg_m * avg_n- avg_m * rn - avg_n * rm)/ \
               (np.sqrt(rm2 + cardn*avg_m**2 - 2*avg_m*rm) * np.sqrt(rn2 + cardn*avg_n**2 - 2*avg_n*rn))