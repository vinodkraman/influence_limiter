# Preprocesses data

from random import shuffle
from time import time

from datetime import date

class Preprocessor:

    def __init__(self, file_path, dataset, label_map=None):
        data = self.read_data(file_path)
        self.user_dict = {}  # user_dict[user_id][item_id] = Rating
        self.item_dict = {}  # item_dict[item_id][user_id] = Rating
        self.time_dict = {}  # for keeping track of time
        self.rating_per_user = {}
        self.rating_per_item = {}
        self.num_ratings = 0

        if label_map:
            self.label_map = label_map
        else:
            self.label_map = {0: -1, 0.5: -1, 
                              1: -1, 1.5: -1,
                              2: -1, 2.5: -1,
                              3: -1, 3.5: 1,
                              4: 1, 4.5: 1,
                              5: 1}

        self.dataset = dataset

        self.process_file(data)

    def read_data(self, file):
        f = open(file, "r", encoding="utf8")
        data = f.read()
        f.close()
        return data


    def return_ratings_list(self):
        """
        Returns list of ratings in the following format:
        [(user, item, rating, timestamp), ...]
        """
        entry_list = []

        for user in self.user_dict.keys():
            for item in self.user_dict[user].keys():
                entry = [user, item, self.user_dict[user][item], self.time_dict[item][user]]
                entry_list.append(entry)
        return entry_list

    def return_item_list(self):
        return self.item_dict.keys()

    def get_heldout_data(self, users=[], no_users=0, min_no=0, percentage=0.20):
        '''
        no_users: number of target users. If None, then
                  just get random no_ratings
        users: List of specific users whose ratings will be removed
        min_no_ratings: min. no. of ratings that each user removed will have
        percentage: % of ratings to remove from matrix. Ignored if no_users != None

        Will return the heldout ratings list
        '''
        heldout_list = []
        if len(users) > 0:
            # For each user in the list
            for user in users:
                if user not in self.rating_per_user.keys():
                    print(f"Error: {user} not in dataset")
                    users.remove(user)
            
        elif no_users > 0:
            # Find no_users with most ratings
            # Note: If not enough, print error
            ordered_users = sorted(self.rating_per_user.items(), key=lambda x: x[1])
            counter = 0
            for user, count in reversed(ordered_users):
                counter += 1
                if counter > no_users:
                    break
                users.append(user)
                

        elif min_no > 0:
            users = [user for user in self.user_dict.keys() if self.rating_per_user[user] >= min_no]
            if len(users) <= 0:
                print(f"Error: no users have {min_no} ratings")

        else:
            all_entries = self.return_ratings_list()
            shuffle(all_entries)
            index = int(percentage*len(all_entries))
            heldout_list = all_entries[0:index]
            for user, item, rating, timestamp in heldout_list:
                del self.item_dict[item][user]
                del self.time_dict[item][user]
                self.rating_per_item[item] -= 1
                del self.user_dict[user][item]
                self.rating_per_user[user] -= 1
                self.num_ratings -= 1
                if self.rating_per_user[user] == 0:
                    del self.rating_per_user[user]
                    del self.user_dict[user]
                if self.rating_per_item[item] == 0:
                    del self.time_dict[item]
                    del self.item_dict[item]
                    del self.rating_per_item[item]

        if len(users) > 0 and len(heldout_list) <= 0:
            for user in users:
                rating_dict = self.user_dict[user]

                for item, rating in rating_dict.items():
                    heldout_list.append((user, item, rating, self.time_dict[item][user]))
                    del self.item_dict[item][user]
                    self.rating_per_item[item] -= 1
                    del self.time_dict[item][user]

                del self.user_dict[user]
                self.num_ratings -= self.rating_per_user[user]
                del self.rating_per_user[user]

        return heldout_list


    def print_stats(self, num=5):
        # Print information about the dataset so far, including
        no_users = len(self.user_dict.keys())
        no_items = len(self.item_dict.keys())

        print(f"No ratings: {self.num_ratings}")
        print(f"No users: {no_users}")
        print(f"No items: {no_items}")
        print(f"Percentage filled: {100*self.num_ratings/(no_users*no_items):.2f}%")
        print(f"Av. rating per user: {self.num_ratings / no_users:.2f}")
        print(f"Av. rating per item: {self.num_ratings / no_items:.2f}")

        ordered_users = sorted(self.rating_per_user.items() ,  key=lambda x: x[1])
        counter = 0
        print(f"{num} Users with least ratings")
        for user, count in ordered_users:
            counter += 1
            if counter > num:
                break
            print(f"Id: {user}, Count: {count}")
            
        counter = 0
        print(f"{num} Users with most ratings")
        for user, count in reversed(ordered_users):
            counter += 1
            if counter > num:
                break
            print(f"Id: {user}, Count: {count}")
            
        ordered_items = sorted(self.rating_per_item.items() ,  key=lambda x: x[1])
        counter = 0
        print(f"{num} Items with least ratings")
        for item, count in ordered_items:
            counter += 1
            if counter > num:
                break
            print(f"Id: {item}, Count: {count}")
            
        counter = 0
        print(f"{num} Items with most ratings")
        for item, count in reversed(ordered_items):
            counter += 1
            if counter > num:
                break
            print(f"Id: {item}, Count: {count}")
            


    def process_file(self, data):
        dataset = self.dataset
        for line in data.split("\n"):
            if dataset == "ciaodvd":
                parts = line.split(",")
            else:
                parts = line.split()
            if len(parts) != 0:
                user_id = int(parts[0])
                item_id = int(parts[1])

                if dataset == "ciaodvd":
                    rating = float(parts[4])
                else:
                    rating = float(parts[2])

                if dataset == "movielens":
                    timestamp = int(parts[3])
                elif dataset == "ciaodvd":
                    timestamp = date.fromisoformat(parts[5]).toordinal()
                else:
                    timestamp = time()

                rating = self.label_map[rating]

                if user_id in self.user_dict.keys():
                    self.user_dict[user_id][item_id] = rating
                    self.rating_per_user[user_id] += 1
                else:
                    self.user_dict[user_id] = {item_id: rating}
                    self.rating_per_user[user_id] = 1

                if item_id in self.item_dict.keys():
                    self.item_dict[item_id][user_id] = rating
                    self.rating_per_item[item_id] += 1
                    self.time_dict[item_id][user_id] = timestamp
                else:
                    self.item_dict[item_id] = {user_id: rating}
                    self.rating_per_item[item_id] = 1
                    self.time_dict[item_id] = {user_id: timestamp}
                self.num_ratings += 1



    def percentage_cleaning(self, percent=0.60, alpha_i=3, alpha_u=2):
        '''
        Iteratively remove users and movies with
        lowest number of ratings until the matrix
        is percent% full.
        alpha: num of movies and users to remove before
        checking if matrix is full enough
        '''
        # NOTE: better way of doing this:
        # Compare if removing lowest item or lowest user will yield closer result to optimal matrix
        
        num_movies = len(self.item_dict.keys())
        num_users = len(self.user_dict.keys())

        # print("num_ratings: {}".format(self.num_ratings))
        # print("ratings needed: {}".format((num_movies)*(num_users)*percent))
        while self.num_ratings < (num_movies)*(num_users)*percent:
            counter = 0
            for user, count in sorted(self.rating_per_user.items() ,  key=lambda x: x[1]):
                if count > 0:
                    counter += 1
                if counter > alpha_u:
                    break

                rating_dict = self.user_dict[user]

                for item, rating in rating_dict.items():
                    del self.item_dict[item][user]
                    self.rating_per_item[item] -= 1
                    del self.time_dict[item][user]

                del self.user_dict[user]
                self.num_ratings -= self.rating_per_user[user]
                del self.rating_per_user[user]

            counter = 0
            for item, count in sorted(self.rating_per_item.items() ,  key=lambda x: x[1]):
                if count > 0:
                    counter += 1 
                if counter > alpha_i:
                    break

                rating_dict = self.item_dict[item]

                for user, rating in rating_dict.items():
                    del self.user_dict[user][item]
                    self.rating_per_user[user] -= 1

                del self.item_dict[item]
                del self.time_dict[item]
                self.num_ratings -= self.rating_per_item[item]
                del self.rating_per_item[item]

            num_movies = len(self.item_dict.keys())
            num_users = len(self.user_dict.keys())




    def threshold_cleaning(self, min_items=3, min_users=3):

        criterion_met = False
        while not criterion_met:
            criterion_met = True
            for user, count in sorted(self.rating_per_user.items() ,  key=lambda x: x[1]):
                if count < min_items:
                    rating_dict = self.user_dict[user]

                    for item, rating in rating_dict.items():
                        del self.item_dict[item][user]
                        self.rating_per_item[item] -= 1
                        del self.time_dict[item][user]

                    del self.user_dict[user]
                    self.num_ratings -= self.rating_per_user[user]
                    del self.rating_per_user[user]
                    criterion_met = False
                else:
                    break

            for item, count in sorted(self.rating_per_item.items() ,  key=lambda x: x[1]):
                if count < min_users:

                    rating_dict = self.item_dict[item]

                    for user, rating in rating_dict.items():
                        del self.user_dict[user][item]
                        self.rating_per_user[user] -= 1

                    del self.item_dict[item]
                    del self.time_dict[item]
                    self.num_ratings -= self.rating_per_item[item]
                    del self.rating_per_item[item]
                    criterion_met = False
                else:
                    break