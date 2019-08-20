# Functions for trust (hasn't really been used for analysis, was for a direction we didn't choose to take)

def build_trust_matrix(recsys, path, dataset="epinions"):

    f = open(path, "r", encoding="utf8")
    data = f.read()
    f.close()

    reps = {}
    for line in data.split("\n"):
        if dataset == "epinions" or dataset == "filmtrust":
            parts = line.split()
        elif dataset == "ciaodvd":
            parts = line.split(",")
        else:
            print("Error: dataset not understood")
            return reps
        
        if len(parts) != 0:
            truster_id = int(parts[0])
            trustee_id = int(parts[1])
            rating = int(parts[2])

            if truster_id in recsys.recsys.user_dict:
                if truster_id not in reps:
                    reps[truster_id] = {}
                if trustee_id in recsys.recsys.user_dict:
                    reps[truster_id][trustee_id] = rating
                # print("Trustee {} not in user dict".format(trustee_id))
            else:
                # print("Truster {} not in user dict".format(truster_id))
                pass

    return reps

def calc_loss(true_val, pred, loss_type):
    if loss_type == "squared":
        return (true_val - pred) ** 2


def calc_metric(true_val, pred, metric):
    label = 1 if pred >= 1 else 0
    if metric == "accuracy":
        return int(label == true_val)


def get_trust_measure(trust, reps, implicit_distrust=False, loss_type="squared", metric_type="accuracy"):

    loss = 0
    acc = 0

    counter = 0
    for truster in reps.keys():
        if truster in trust or implicit_distrust:
            for trustee in reps[truster].keys():
                pred = min(1, reps[truster][trustee])
                if trustee in trust[truster]:
                    true_val = trust[truster][trustee]
                elif implicit_distrust:
                    true_val = 0
                else:
                    continue
                curr_loss = calc_loss(true_val, pred, loss_type)
                curr_metric = calc_metric(true_val, pred, metric_type)
                loss += (1/(1 + counter))*(curr_loss - loss)
                acc += (1 / (1 +counter))*(curr_metric - acc)
                counter += 1

    return loss, acc, counter