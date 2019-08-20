import numpy as np
import matplotlib.pyplot as plt

def run_experiment(recsys, heldout_data, attack_id={}, t_items={}):
    '''
    Given a prebuilt, initialized recsys
    - recys: prebuilt, initialized recommender system
    - heldout: list of ordered ratings (user, item, rating)
    - attack_id: id's of attacking users
    Sequentially adds each rating
    Returns il_loss, nil_loss, il_acc, nil_acc, recsys
    NOTE: losses are per round, accuracy is average
    '''
    il_losses = []
    il_accs = []
    nil_losses = []
    nil_accs = []
    accs_diff = []
    losses_diff = []

    av_il_loss = 0
    av_il_acc = 0
    av_nil_loss = 0
    av_nil_acc = 0

    av_loss_diff = 0
    av_acc_diff = 0


    # If no attackers, counter is the same as k
    # Otherwise, k will be used for computing average
    # counter will be used for placeholding
    k = 0

    t_nil_acc = 0
    t_nil_accs = []
    t_k = 0
    t_il_acc = 0
    t_il_accs = []

    for idx, entry in enumerate(heldout_data):
        user = entry[0]
        item = entry[1]
        rating = entry[2]

        q_t, q = recsys.recommend(user, item, False)

        il_loss, il_acc, nil_loss, nil_acc = recsys.receive_label(user, item ,rating)

        if user not in attack_id:
            # User is not attacker, should include loss/accuracy here
            av_nil_acc += (1/(k + 1))*(nil_acc - av_nil_acc)
            av_il_acc += (1/(k + 1))*(il_acc - av_il_acc)
            # av_nil_loss += (1/(k + 1))*(nil_loss - av_nil_loss)
            # av_il_loss += (1/(k + 1))*(il_loss - av_il_loss)
            acc_diff = nil_acc - il_acc
            loss_diff = nil_loss - il_loss
            # av_acc_diff += (1/(k + 1))*(acc_diff - av_acc_diff)
            av_loss_diff += (1/(k + 1))*(loss_diff - av_loss_diff)

            nil_accs.append(av_nil_acc)
            il_accs.append(av_il_acc)

            # nil_losses.append(nil_loss)
            # il_losses.append(il_loss)

            # accs_diff.append(av_acc_diff)
            losses_diff.append(loss_diff)
            k += 1

            if item in t_items:
                t_nil_acc += (1/(t_k + 1))*(nil_acc - t_nil_acc)
                t_il_acc += (1/(t_k + 1))*(il_acc - t_il_acc)
                t_k += 1
                t_nil_accs.append(t_nil_acc)
                t_il_accs.append(t_il_acc)

    t_nil_accs = np.array(t_nil_accs)
    t_il_accs = np.array(t_il_accs)
    il_accs = np.array(il_accs)
    nil_accs = np.array(nil_accs)
    losses_diff = np.array(losses_diff)

    return  il_accs, nil_accs, t_il_accs, t_nil_accs, losses_diff, recsys



def make_plots(fig_path, losses_diff, il_accs, nil_accs, t_il_accs, t_nil_accs, attack_ids, recsys):

    x = np.arange(len(losses_diff))
    # Loss Graph
    plt.figure(1)
    plt.scatter(x, losses_diff, s=2, color='g', label='NIL - IL')
    plt.xlabel("Item No.")
    plt.ylabel("Loss Difference per Item Recommendation")
    plt.legend()
    plt.title("Recommendation Loss")
    plt.savefig(fig_path + "loss.png")
    plt.clf()

    # Acc Graph
    plt.figure(2)
    plt.plot(x, il_accs, color='r', label='IL')
    plt.plot(x, nil_accs, color='b', label='Non-IL')
    plt.xlabel("Item No.")
    plt.ylabel("Average Accuracy")
    plt.legend()
    plt.title("Recommendation Accuracy")
    plt.savefig(fig_path + "acc.png")
    plt.clf()


    # Plot Reputations
    reps, impacts = recsys.return_rep_impacts()

    sorted_users = sorted(impacts.items(),  key=lambda x: x[1][-1])
    sorted_users = [x[0] for x in sorted_users]
    tracked_users = []
    tracked_users.extend(sorted_users[:2])
    tracked_users.extend(sorted_users[-2:])
    tracked_users.append(sorted_users[len(sorted_users) // 2])

    plt.figure(3)
    x_axis = np.arange(len(reps[tracked_users[0]]) )
    colors = ['r', 'b', 'y', 'g', 'c']
    for idx, user in enumerate(tracked_users):
        plt.plot(x_axis, reps[user], color=colors[idx % len(colors)], label='User {}'.format(user)) 
    plt.xlabel("Item No.")
    plt.ylabel("Cumulative Reputation")
    plt.legend()
    plt.title("Users' Reputation")
    plt.savefig(fig_path + "reps.png")
    plt.clf()


    plt.figure(4)
    x_axis = np.arange(len(impacts[tracked_users[0]]) )
    colors = ['r', 'b', 'y', 'g', 'c']
    for idx, user in enumerate(tracked_users):
        plt.plot(x_axis, impacts[user], color=colors[idx % len(colors)], label='User {}'.format(user)) 
    plt.xlabel("Item No.")
    plt.ylabel("Cumulative Impact")
    plt.legend()
    plt.title("Users' Impact")
    plt.savefig(fig_path + "impact.png")
    plt.clf()

    # Plot Shill reputations

    tracked_users = [i for i in attack_ids]

    plt.figure(5)
    x_axis = np.arange(len(reps[tracked_users[0]]) )
    colors = ['r', 'b', 'y', 'g', 'c']

    final_reps = np.zeros(len(x_axis))
    for idx, user in enumerate(tracked_users):
        # plt.plot(x_axis, reps[user], color=colors[idx % len(colors)], label='rep user {}'.format(user)) 
        final_reps += reps[user]

    plt.plot(x_axis, final_reps, color=colors[idx % len(colors)], label='All shills') 
    plt.xlabel("Item No.")
    plt.ylabel("Cumulative Reputation")
    plt.legend()
    plt.title("Shills' Reputations")
    plt.savefig(fig_path + "shills_reps.png")
    plt.clf()


    plt.figure(6)
    x_axis = np.arange(len(impacts[tracked_users[0]]) )
    colors = ['r', 'b', 'y', 'g', 'c']
    final_impacts = np.zeros(len(x_axis))
    for idx, user in enumerate(tracked_users):
        # plt.plot(x_axis, reps[user], color=colors[idx % len(colors)], label='rep user {}'.format(user)) 
        final_impacts += impacts[user]

    plt.plot(x_axis, final_impacts, color=colors[idx % len(colors)], label='All shills') 
    plt.xlabel("Item No.")
    plt.ylabel("Cumulative Impact")
    plt.legend()
    plt.title("Shills' Impact")
    plt.savefig(fig_path + "shills_impact.png")
    plt.clf()


    # Plot Target Item Accuracy
    # Acc Graph
    plt.figure(2)
    x = np.arange(len(t_il_accs))
    plt.plot(x, t_il_accs, color='r', label='IL')
    plt.plot(x, t_nil_accs, color='b', label='Non-IL')
    plt.xlabel("Targeted Item No.")
    plt.ylabel("Average Accuracy")
    plt.legend()
    plt.title("Recommendation Accuracy for Targeted Items")
    plt.savefig(fig_path + "t_acc.png")
    plt.clf()
