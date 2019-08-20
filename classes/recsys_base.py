from time import time

class RecSysBase:
    # Base class for RecSys black-box algorithms

    def __init__(self, entry_list):
        # Steps for initializing data
        self.user_dict = {}  # user_dict[user_id][item_id] = Rating
        self.item_dict = {}  # item_dict[item_id][user_id] = Rating
        self.time_dict = {}  # time_dict[item_id][user_id] = Timestamp
        self.rating_per_user = {}
        self.rating_per_item = {}
        self.num_ratings = 0
        self.add_ratings(entry_list)


    def set_up(self, user, item):
        # Given a target (user, item), preps to sequentially
        # remove all ratings for that item and start
        # q-generating process
        # if user not in self.user_dict:
        #     print("Error: user not seen")
        #     return
        # if item not in self.item_dict:
        #     print("Error: item not seen")
        #     return

        # pass
        # return [(curr_user, rating) for curr_user, rating in self.item_dict[item]]
        return

    def add_new_user(self, user):
        self.user_dict[user] = {}
        self.rating_per_user[user] = 0

    def add_new_item(self, item):
        self.item_dict[item] = {}
        self.time_dict[item] = {}
        self.rating_per_item[item] = 0

    def add_sequential_rating(self, rater, item, rating):
        # Given (user, item, rating), incorporates it and updates any
        # necessary data structures
        pass


    def make_recommendation(self, user, item):
        # Given a (user,item) pair, it returns a predicted LABEL
        # if user not in self.user_dict:
        #     print("Error: user not seen")
        #     return
        # if item not in self.item_dict:
        #     print("Error: item not seen")
        #     return
        return


    def prior(self, user, item):
        # Give prior prediction
        return 0.0


    def receive_rating(self, user, item, rating, timestamp=None):
        # Given a NEW rating, adds it to all data structures

        if timestamp == None:
            timestamp = time()

        if user in self.user_dict.keys():
            self.user_dict[user][item] = rating
            self.rating_per_user[user] += 1
        else:
            self.user_dict[user] = {item: rating}
            self.rating_per_user[user] = 1

        if item in self.item_dict.keys():
            self.item_dict[item][user] = rating
            self.rating_per_item[item] += 1
            self.time_dict[item][user] = time()
        else:
            self.item_dict[item] = {user: rating}
            self.rating_per_item[item] = 1
            self.time_dict[item] = {user: time()}

        pass

    
    def get_rated_user_list(self, item):
        user_list = []
        for user in self.item_dict[item].keys():
            user_list.append((user, self.time_dict[item][user]))
        user_list = sorted(user_list, key=lambda x:x[1])
        user_list = [i[0] for i in user_list]
        return user_list


    def add_ratings(self, entry_list):
        '''
        Takes in a list of (user, item, rating, timestamp)
        and adds them to the dictionaries.

        NOTE: Must update any invariants within that class
        '''
        for entry in entry_list:
            user = entry[0]
            item = entry[1]
            rating = entry[2]
            timestamp = entry[3]
            self.receive_rating(user, item, rating, timestamp)