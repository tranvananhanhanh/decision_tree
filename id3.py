import numpy as np
import math

# Hàm tính entropy
def entropy(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# Hàm tính Information Gain
def information_gain(data, labels, feature_index):
    feature_values = np.unique(data[:, feature_index])
    total_entropy = entropy(labels)
    weighted_entropy = 0

    for value in feature_values:
        subset_indices = np.where(data[:, feature_index] == value)
        subset_labels = labels[subset_indices]
        subset_entropy = entropy(subset_labels)
        subset_weight = len(subset_labels) / len(labels)
        weighted_entropy += subset_weight * subset_entropy

    information_gain = total_entropy - weighted_entropy
    return information_gain

# Hàm chọn thuộc tính tốt nhất để chia dữ liệu
def choose_best_feature(data, labels):
    num_features = data.shape[1] - 1
    best_feature = 0
    best_information_gain = 0

    for feature_index in range(num_features):
        information_gain_value = information_gain(data, labels, feature_index)
        if information_gain_value > best_information_gain:
            best_information_gain = information_gain_value
            best_feature = feature_index

    return best_feature

# Hàm xây dựng cây quyết định
def build_decision_tree(data, labels, feature_names, parent_label=None):
    # Kiểm tra điều kiện dừng
    if len(np.unique(labels)) == 1:
        return labels[0]
    if data.shape[1] == 1:
        unique_labels, counts = np.unique(labels, return_counts=True)
        return unique_labels[np.argmax(counts)]

    # Chọn thuộc tính tốt nhất để chia dữ liệu
    best_feature = choose_best_feature(data, labels)
    best_feature_name = feature_names[best_feature]

    # Xây dựng cây quyết định
    decision_tree = {best_feature_name: {}}

    feature_values = np.unique(data[:, best_feature])

    for value in feature_values:
        value_indices = np.where(data[:, best_feature] == value)
        subset_data = data[value_indices][:, 1:]
        subset_labels = labels[value_indices]
        subset_feature_names = np.delete(feature_names, best_feature)
        decision_tree[best_feature_name][value] = build_decision_tree(subset_data, subset_labels, subset_feature_names)

    return decision_tree

# Dữ liệu huấn luyện
train_data = np.array([
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rainy', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rainy', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rainy', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rainy', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rainy', 'Mild', 'High', 'Strong', 'No']
])

# Tên thuộc tính
feature_names = np.array(['Outlook', 'Temperature', 'Humidity', 'Wind'])

# Tạo cây quyết định
decision_tree = build_decision_tree(train_data, train_data[:, -1], feature_names)

# In cây quyết định
print(decision_tree)