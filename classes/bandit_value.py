# UNFINISHED CLASS FOR A NEW TYPE OF RECSYS BLACK BOX

import numpy as np

from .recsys_base import *


class BanditValue(RecSysBase):

    def __init__(self, entry_list, params={"corr": "pearson", "strat": "top_k", "param": 5}):
        # Steps for initializing data

        super().__init__(entry_list)


        '''
        Off-line training:
        - Create user features and item features
        - Create sequential training examples


        Pending helper functions:
        - make_user_feats(user_id)
        - update_user_feats(user_id, item_id, rating)
        - make_item_feats(item_id)
        - update_item_feats(user_id, item_id, rating)
        - reset_item_feats(item_id)

        '''


    def set_up(self, user, item):
        super().set_up(user, item)

        user_list = self._get_rated_user_list(item)
        self.t_user = user

        if user not in self.user_dict.keys():
            self.add_new_user(user)
            print("added new user")

        return [(curr_user, self.item_dict[item][curr_user]) for curr_user in user_list]


    def add_new_user(self, user):
        super().add_new_user(user)


    def add_ratings(self, entry_list):
        super().add_ratings(entry_list)


    def add_sequential_rating(self, rater, item, rating):
        # Given (user, item, rating), incorporates it and updates any


    def prior(self, user, item):
        # Give prior prediction
        return 0.0


    def make_recommendation(self, user, item):

        super().make_recommendation(user, item)

        if user not in self.user_dict.keys():
            return self.prior(user, item)



    def receive_rating(self, user, item, rating, timestamp=None):
        '''
        Adds in new rating to dictionaries.
        Updates reputations if applicable
        '''

        super().receive_rating(user, item, rating, timestamp)