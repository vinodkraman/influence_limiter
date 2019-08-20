
from classes.preprocessor import Preprocessor
from classes.il_recsys import ILRecSys

import numpy as np
import sys
import matplotlib.pyplot as plt

from functions.attack import *
from functions.utility import *
from random import shuffle

import yaml
import os
from copy import deepcopy

# will use this config file to set settings
YAML_FILE = "./config/shills.yml"
with open(YAML_FILE, 'r') as file:
    y_d = yaml.load(file)

dataset = y_d["dataset"]
data_file = y_d["paths"][ dataset]["ratings"]

print("----Dataset Details----")
print(f"Dataset: {dataset}")

# Preprocess data
data_class = Preprocessor(data_file, dataset)

if y_d["clean"]["type_of"] == "percent":
    # Cleans using percentage cleaning (matrix has to be x% filled)
    alpha_i = y_d["clean"]["alpha_i"]
    alpha_u = y_d["clean"]["alpha_u"]
    data_class.percentage_cleaning(percent=y_d["clean"]["percent"], alpha_i=alpha_i, alpha_u=alpha_u)
    print("Cleaning Strategy: Percentage")
    print(f'Cleaning Percentage: {y_d["clean"]["percent"]}')
    print(f"Cleaning Params: {alpha_i} {alpha_u}")
elif y_d["clean"]["type_of"] == "threshold":
    # Cleans until all items meet the min number of users and viceversa
    min_items = y_d["clean"]["min_items"]
    min_users = y_d["clean"]["min_users"]
    data_class.threshold_cleaning(min_items=min_items, min_users=min_users)
    print("Cleaning Strategy: Threshold")
    print(f"Cleaning Params: {min_items} {min_users}")
else:
    print("ERROR: cleaning not recognized, won't clean")

data_class.print_stats()


print("----Experiment Details----")

c = y_d["c_"]
n = y_d["n_"]
t_user = y_d["tracked_user"]
print(f"Target User: {t_user}")

# Separate data into initialization and heldout
# Heldout is used to simulate new ratings, intialization is initial matrix
heldout_data = data_class.get_heldout_data(users=[t_user], no_users=0, min_no=0, percentage=0.0)
orig_entry_list = data_class.return_ratings_list()

# Number of targeted items and targeted items
n_targets = y_d["attack"]["no_targets"]
target_items = {entry[1] for entry in heldout_data[-n_targets:] }

# Number of ratings random/determ clones will make
no_items = y_d["attack"]["no_ratings"]

# Rating for targeted items
t_rating = y_d["attack"]["t_rating"]

# Users to clone for clone clones
cloned_users = y_d["attack"]["cloned"]

# No of attackers
no_clones = y_d["attack"]["no_attackers"]

il_graph = []
nil_graph = []
t_il_graph = []
t_nil_graph = []

c = 20


# Initialize recommender system
recsys = ILRecSys(orig_entry_list, black_box="knn", params={"corr": "pearson", "strat": "top_k", "param": 10}, 
                  n=n, c=c, loss_type="squared", tracked_user=t_user)

# Add attack
attack_ratings = []
attack_ids = set()
print("no clones: {}".format(no_clones))
for i in range(no_clones):
    if y_d["attack"]["type_of"] == "clone":
        attack = create_clone_attack(recsys, copied_user=cloned_users[i % len(cloned_users)], 
                                     clone_id=-i, t_items=target_items, t_rating=t_rating)
    elif y_d["attack"]["type_of"] == "random":
        attack = create_random_attack(recsys, no_items=no_items, clone_id=-i, t_items=target_items, t_rating=t_rating)
    elif y_d["attack"]["type_of"] == "determ":
        attack = create_determ_attack(recsys, no_items=no_items, clone_id=-i, t_items=target_items, t_rating=t_rating)
    else:
        print("Error, attack type not recognized. won't attack")
        break
    attack_ratings.extend(attack)
    attack_ids.add(-i)

# Establish order for ratings in system
place_attack = y_d["attack"]["place_attack"]
print(f"Attackers Place: {place_attack}")
if place_attack == "first":
    entry_list = [ [entry[0], entry[1], entry[2], time()] for entry in orig_entry_list ]
    entry_list.extend(attack_ratings)
    entry_list.reverse()
elif place_attack == "last":
    entry_list = deepcopy(orig_entry_list)
    entry_list.extend(attack_ratings)
elif place_attack == "random":
    entry_list = deepcopy(orig_entry_list)
    entry_list.extend(attack_ratings) 
    shuffle(entry_list)
    entry_list = [ [entry[0], entry[1], entry[2], time()] for entry in entry_list ]

# Establish order for heldout ratings
place_target = y_d["attack"]["place_target"]
if place_target == "first":
    # Attackers will go before others w.r.t. timestamps
    # Has to delete all timestamps in normal ratings
    heldout_data.reverse()
elif place_target == "last":
    # Don't have to do anything
    pass
elif place_target == "random":
    # Has to extend, then recreate 
    shuffle(heldout_data)

# Remake RecSys with attack ids
recsys = ILRecSys(entry_list, black_box="knn", params={"corr": "pearson", "strat": "top_k", "param": 30}, 
                  n=n, c=c, loss_type="squared", tracked_user=t_user)

# Run experiment once
il_accs, nil_accs, t_il_accs, t_nil_accs, losses_diff, recsys  = run_experiment(recsys, heldout_data, attack_id=attack_ids, t_items=target_items)
'''
For more details on these data structures, i highly recommend checking the function run_exmperiment in functions/utility.py

il_accs: Influence limited average accuracy at each round on all items
nil_accs: Non-Influence limited average accuracy at each round
t_il_accs: Influence limited average accuracy at each round on target items
t_nil_accs: Non-Influence limited average accuracy at each round on target items
losses_diff: Differences in loss between nil and il (nil - il) per round (not average)
recsys: recsys object with heldout data incorporated into it
'''
print(il_accs)