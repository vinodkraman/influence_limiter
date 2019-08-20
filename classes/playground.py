import numpy as np
import matplotlib.pyplot as plt

signal_label = np.array([[.6, .5], [.4, .5]])
label = np.array([.4, .6])
print(np.array(signal_label))
print(label)
print(signal_label * label)
print(np.matmul(signal_label, label))
print(np.transpose(signal_label * label))
ans = np.divide(np.transpose(signal_label * label), np.matmul(signal_label, label))
print(ans)
print(ans[:,1].tolist())
label_signal = np.array([[.444444444, .34782609],[0.55555555,.65217391]])
print(np.matmul(signal_label, label_signal))
print(np.matmul(signal_label, ans[:,1]))



matrix = []
p = 1/5
prob_vec = (p * np.ones(5)).tolist()
target_label_space = {1,2,3,4}
for label in target_label_space:
    matrix.append(prob_vec)

def generate_item_conditional_signal_priors(item_list, target_label_space, signal_space):
    item_conditional_signal_priors = {}
    signal_space_length = len(signal_space)

    for item in item_list:
        matrix = []
        for label in target_label_space:
            prob_vec = np.random.random(signal_space_length)
            prob_vec /= prob_vec.sum()
            matrix.append(prob_vec)
        matrix = np.transpose(matrix)
        item_conditional_signal_priors[item] = matrix
    
    return item_conditional_signal_priors

item_list = [1, 2, 3, 4, 5]
target_label_space = {1, 2, 3, 4}
signal_space = {1, 2, 3}

def generate_item_target_label_priors(item_list, target_label_space):
    item_to_target_label_prior = {}
    length = len(target_label_space)

    for item in item_list:
        p = 1/length
        prob_vec = (p * np.ones(length)).tolist()
        item_to_target_label_prior[item] = prob_vec

    return item_to_target_label_prior
    
print(generate_item_conditional_signal_priors(item_list, target_label_space, signal_space))