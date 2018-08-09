
import math
import numpy as np

def calc_shannonent(dataset):
    num_entries = len(dataset)
    labelcounts = {}
    for featureVec in dataset:
        current_label = featureVec[-1]
        if current_label not in labelcounts.keys():
            labelcounts[current_label] = 0
        labelcounts[current_label] += 1
    shannonent = 0.0
    for key in labelcounts:
        prob = float(labelcounts[key]) / num_entries
        shannonent -= prob * math.log(prob, 2)

    return shannonent


#test
def create_dataset():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]

    labels = ['no surfacing', 'flippers']
    return dataset, labels

def split_dataset(dataset, axis, value):
    ret_dataset = []
    for feature_vec in dataset:
        if feature_vec[axis] == value:
            reduce_feature_vec = feature_vec[:axis]
            reduce_feature_vec.extend(feature_vec[axis + 1 :])
            ret_dataset.append(reduce_feature_vec)
    return ret_dataset

def choose_best_feature_2_split(dataset):
    num_features = len(dataset[0]) - 1
    base_shanonent = calc_shannonent(dataset)
    best_feature = -1
    best_info_gain = 0.0

    for i in range(num_features):
        feat_list = [example[i] for example in dataset]
        unique_vals = set(feat_list)
        new_entropy = 0.0
        for value in unique_vals:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset) / float(len(dataset))
            new_entropy += prob * calc_shannonent(sub_dataset) #类似于熵的期望？
            # new_entropy +=  calc_shannonent(sub_dataset)
        info_gain = base_shanonent - new_entropy
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_voting(class_list):
    import operator

    class_count = {}
    for vote in class_list:
        if vote not in class_list.keys():
            class_count[vote] = 0
            class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(), key = operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

def create_tree(dataset, labels):
    class_list = [example[-1] for example in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    if len(dataset[0]) == 1:
        return majority_voting(class_list)

    best_feature = choose_best_feature_2_split(dataset)
    best_feature_label = labels[best_feature]
    myTree = {best_feature_label:{}}
    del labels[best_feature]
    feature_value = [example[best_feature] for example in dataset]
    unique_values = set(feature_value)
    for value in unique_values:
        sublabels = labels[:]
        myTree[best_feature_label][value] = create_tree(split_dataset(dataset, best_feature, value), sublabels)

    return myTree




dataset, labels = create_dataset()
ret = split_dataset(dataset, 1, 1)
shannon = calc_shannonent(dataset)
best_feature = choose_best_feature_2_split(dataset)
tree = create_tree(dataset, labels)
a = 0