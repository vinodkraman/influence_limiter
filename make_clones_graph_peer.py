
from classes.preprocessor import Preprocessor
from classes.il_recsys import ILRecSys
from classes.peer_il_recsys import PeerILRecSys

import numpy as np
import sys
import matplotlib.pyplot as plt

from functions.attack import *
from functions.utility import *
from random import shuffle

import yaml
import os
from copy import deepcopy

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
    alpha_i = y_d["clean"]["alpha_i"]
    alpha_u = y_d["clean"]["alpha_u"]
    data_class.percentage_cleaning(percent=y_d["clean"]["percent"], alpha_i=alpha_i, alpha_u=alpha_u)
    print("Cleaning Strategy: Percentage")
    print(f'Cleaning Percentage: {y_d["clean"]["percent"]}')
    print(f"Cleaning Params: {alpha_i} {alpha_u}")
elif y_d["clean"]["type_of"] == "threshold":
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
heldout_data = data_class.get_heldout_data(users=[t_user], no_users=0, min_no=0, percentage=0.0)
orig_entry_list = data_class.return_ratings_list()

n_targets = y_d["attack"]["no_targets"]
target_items = {entry[1] for entry in heldout_data[-n_targets:] }
no_items = y_d["attack"]["no_ratings"]
t_rating = y_d["attack"]["t_rating"]
cloned_users = y_d["attack"]["cloned"]


il_graph = []
nil_graph = []
t_il_graph = []
t_nil_graph = []
sum_clone_impact = []
impact_bounds = []
no_clones = [5, 10, 20]
c = 20

item_list = list(data_class.return_item_list())
item_target_label_prior = generate_item_target_label_priors(item_list, {-1, 1})
item_conditional_signal_prior = generate_item_conditional_signal_priors(item_list, {-1, 1}, {-1, 1})

print(item_target_label_prior)
print(item_conditional_signal_prior)
for run in range(1):
    # shuffle(heldout_data)
    curr_il_graph = []
    curr_nil_graph = []
    curr_t_il_graph = []
    curr_t_nil_graph = []
    for clones in no_clones:

        # Initialize recommender system
        recsys = PeerILRecSys(orig_entry_list, black_box="knn", params={"corr": "pearson", "strat": "top_k", "param": 10}, 
        item_conditional_signal_prior=item_conditional_signal_prior, item_target_label_prior=item_target_label_prior, 
        n=n, c=c, loss_type="squared", tracked_user=t_user)

        # Add attack
        attack_ratings = []
        attack_ids = set()
        print("no clones: {}".format(clones))
        for i in range(clones):
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
        recsys = PeerILRecSys(entry_list, black_box="knn", params={"corr": "pearson", "strat": "top_k", "param": 10}, 
        item_conditional_signal_prior=item_conditional_signal_prior, item_target_label_prior=item_target_label_prior, 
        n=n, c=c, loss_type="squared", tracked_user=t_user)

        # Run experiment once
        il_accs, nil_accs, t_il_accs, t_nil_accs, losses_diff, recsys  = run_experiment(recsys, heldout_data, attack_id=attack_ids, t_items=target_items)
        curr_il_graph.append(il_accs[-1])
        curr_nil_graph.append(nil_accs[-1])
        curr_t_il_graph.append(t_il_accs[-1])
        curr_t_nil_graph.append(t_nil_accs[-1])

        final_imp_sybil = []
        tracked_reps, tracked_impact = recsys.return_rep_impacts()

        for key, value in tracked_impact.items():
            if key <= 0:
                final_imp_sybil.append(value[-1])
        
        sum_clone_impact.append(np.sum(final_imp_sybil))
        impact_bounds.append(-1 * clones * np.exp(-np.log(c*n)))
   

    if len(il_graph) > 0:
        il_graph += (1 / (run + 1))*(curr_il_graph - il_graph)
        nil_graph += (1 / (run + 1))*(curr_nil_graph - nil_graph)
        t_il_graph += (1 / (run + 1))*(curr_t_il_graph - t_il_graph)
        t_nil_graph += (1 / (run + 1))*(curr_t_nil_graph - t_nil_graph)
    else:
        il_graph = curr_il_graph
        nil_graph = curr_nil_graph
        t_il_graph = curr_t_il_graph
        t_nil_graph = curr_t_nil_graph

fig_path = "./figures/" + y_d["attack"]["type_of"] + "/"

if not os.path.exists(fig_path):
    os.makedirs(fig_path)

# Plot Target Item Accuracy
# Acc Graph
plt.figure(2)
x = np.asarray(no_clones)
plt.plot(x, il_graph, color='r', label='IL')
plt.plot(x, nil_graph, color='b', label='Non-IL')
plt.xlabel("Number of Clones")
plt.ylabel("Average Accuracy")
plt.legend()
plt.title("Recommendation Accuracy")
plt.savefig(fig_path + "num_attack_all.png")
plt.clf()


# Plot Target Item Accuracy
# Acc Graph
plt.figure(2)
x = np.asarray(no_clones)
plt.plot(x, t_il_graph, color='r', label='IL')
plt.plot(x, t_nil_graph, color='b', label='Non-IL')
plt.xlabel("Number of Clones")
plt.ylabel("Average Accuracy")
plt.legend()
plt.title("Recommendation Accuracy for Targeted Items")
plt.savefig(fig_path + "num_attack_target.png")
plt.clf()

plt.figure(3)
x = np.asarray(no_clones)
plt.plot(x, sum_clone_impact, color='r', label='actual')
plt.plot(x, impact_bounds, color='b', label='bound')
plt.xlabel("Number of Clones")
plt.ylabel("Damage")
plt.legend()
plt.title("Recommendation Limited Damage Bound")
plt.savefig(fig_path + "limited_damage2.png")
plt.clf()