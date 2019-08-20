
from classes.preprocessor import Preprocessor
from classes.il_recsys import ILRecSys

import numpy as np
import sys
import matplotlib.pyplot as plt

from functions.trust import *
from functions.utility import *
from random import shuffle


# file_path = "./data/ciaodvd/movie-ratings.txt"
# trust_path = "./data/ciaodvd/trusts.txt"
# dataset = "ciaodvd"

file_path = "./data/epinions/ratings_data.txt"
trust_path = "./data/epinions/trust_data.txt"
dataset = "epinions"

# file_path = "./data/filmtrust/ratings.txt"
# trust_path = "./data/filmtrust/trust.txt"
# dataset = "filmtrust"


# Preprocess data
data_class = Preprocessor(file_path, dataset)

print("gonna percentage clean")
data_class.print_stats()
# data_class.percentage_cleaning(percent=0.10, alpha_i=3, alpha_u=2)
data_class.threshold_cleaning(min_items=20, min_users=20)
data_class.print_stats()

c = 1
n = 5

entry_list = data_class.return_ratings_list()

# Initialize recommender system
recsys = ILRecSys([], black_box="knn", params={"corr": "pearson", "strat": "top_k", "param": 30}, 
                  n=n, c=c, loss_type="squared")

heldout_data = entry_list
shuffle(heldout_data)

# Run experiment once
for entry in heldout_data:
    user = entry[0]
    item = entry[1]
    rating = entry[2]
    q_t, q = recsys.recommend(user, item, False)
    il_loss, il_acc, nil_loss, nil_acc = recsys.receive_label(user, item ,rating)

# Compare with original trust stuff
trust_matrix = build_trust_matrix(recsys, path=trust_path, dataset=dataset)

reps = recsys.reps

loss, accuracy, count = get_trust_measure(trust_matrix, reps, implicit_distrust=False, 
                                          loss_type="squared", metric_type="accuracy")


print("Reps average acc: {}".format(accuracy))
print("Reps average loss: {}".format(loss))
print("Reps compared: {}".format(count))
