import numpy as np
from time import time


def create_clone_attack(recsys, copied_user, clone_id, t_items, t_rating=-1):
    if copied_user not in recsys.recsys.user_dict.keys():
        print(f"Error: {copied_user} not found, will use random user")
        copied_user = np.random.choice(list(recsys.recsys.user_dict.keys()))
    if clone_id in recsys.recsys.user_dict.keys():
        print(f"Error: clone id {clone_id} alread exists in system")
        return []
    entry_list = []
    rated_targets = set()
    for item, rating in recsys.recsys.user_dict[copied_user].items():
        if item in t_items:
            entry = [clone_id, item, t_rating, time()]
            rated_targets.add(item)
        else:
            entry = [clone_id, item, rating, time()]
        entry_list.append(entry)

    for item in t_items:
        if item not in rated_targets:
            entry = [clone_id, item, rating, time()]
            entry_list.append(entry)

    return entry_list



def create_random_attack(recsys, no_items, clone_id, t_items, t_rating=-1):
    if no_items <= 0:
        print(f"Error: {no_items} <= 0. Has to be at least 1")
        return []
    if no_items > len(recsys.recsys.item_dict.keys()):
        print(f"{no_item} > num of items in recsys. Will default to that")
        no_items = len(recsys.recsys.item_dict.keys())
    if clone_id in recsys.recsys.user_dict.keys():
        print(f"Error: clone id {clone_id} alread exists in system")
        return []

    indices = np.random.choice(list(recsys.recsys.item_dict.keys()), no_items - 1, replace=False)
    ratings = np.random.choice([-1, 1], no_items - 1)
    entry_list = [[clone_id, item, rating, time()] for item, rating in zip(indices, ratings)]
    entry_list.extend([[clone_id, t_item, t_rating, time()] for t_item in t_items])
    return entry_list



def create_determ_attack(recsys, no_items, clone_id, t_items, t_rating=-1):
    if no_items <= 0:
        print(f"Error: {no_items} <= 0. Has to be at least 1")
        return []
    if no_items > len(recsys.recsys.item_dict.keys()):
        print(f"{no_item} > num of items in recsys. Will default to that")
        no_items = len(recsys.recsys.item_dict.keys())
    if clone_id in recsys.recsys.user_dict.keys():
        print(f"Error: clone id {clone_id} alread exists in system")
        return []

    indices = np.random.choice(list(recsys.recsys.item_dict.keys()), no_items - 1, replace=False)
    opposite_r = 1 if t_rating == -1 else -1
    entry_list = [[clone_id, item, opposite_r, time()] for item in indices]
    entry_list.extend([[clone_id, t_item, t_rating, time()] for t_item in t_items])
    return entry_list


def create_mimic_attack(heldout_data, no_clones, clone_ids, t_items, t_rating=-1):

    new_heldout = []

    for entry in heldout_data:
        user = entry[0]
        item = entry[1]
        rating = entry[2]

        subseq = []

        if item in t_items:
            clone_rating = t_rating
        else:
            clone_rating = rating

        subseq = [(clone_id, item, clone_rating) for clone_id in clone_ids]

        if item in t_items:
            new_heldout.extend(subseq)
            new_heldout.append(entry)
        else:
            new_heldout.append(entry)
            new_heldout.extend(subseq)

    return new_heldout
