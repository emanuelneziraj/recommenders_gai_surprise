from surprise import Dataset, Reader, KNNBasic, SVD, NMF
from surprise.model_selection import KFold
from collections import defaultdict
import numpy as np

# Path to the data file
file_path = '../ml-latest-small/ratings.csv'

# Reader to read the file into Surprise
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale=(0.5, 5))

# Loading the data
data = Dataset.load_from_file(file_path, reader=reader)

# Initialization of algorithms
algo_knn = KNNBasic()
algo_svd = SVD()
algo_nmf = NMF()

# List of algorithms
algorithms = {'KNN': algo_knn, 'SVD': algo_svd, 'NMF': algo_nmf}

# Store results for the models
results = {model: {k: {'Precision': [], 'Recall': [], 'F1': []} for k in [10, 20, 50]} for model in algorithms}


# Definition of the function to calculate Precision, Recall, and F1-Score
def precision_recall_at_k(predictions, k, threshold):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    precision_avg = sum(prec for prec in precisions.values()) / len(precisions) if precisions else 0
    recall_avg = sum(rec for rec in recalls.values()) / len(recalls) if recalls else 0
    f1_score = 2 * (precision_avg * recall_avg) / (precision_avg + recall_avg) if (
                                                                                              precision_avg + recall_avg) != 0 else 0

    return precision_avg, recall_avg, f1_score


# Cross-validation and evaluation
kf = KFold(n_splits=5)
for model_name, algorithm in algorithms.items():
    for trainset, testset in kf.split(data):
        algorithm.fit(trainset)
        predictions = algorithm.test(testset)
        for k in [10, 20, 50]:
            precision, recall, f1 = precision_recall_at_k(predictions, k=k, threshold=3.5)
            results[model_name][k]['Precision'].append(precision)
            results[model_name][k]['Recall'].append(recall)
            results[model_name][k]['F1'].append(f1)


# Calculate the average values for each model and each k-value
def calculate_averages(model_name):
    averages = {k: {} for k in [10, 20, 50]}
    for k in averages:
        for metric in ['Precision', 'Recall', 'F1']:
            averages[k][metric] = np.mean(results[model_name][k][metric])
    return averages


# Output of average values
# Output of average values with four decimal places
output_file_path_2 = "metrics_results.txt"
with open(output_file_path_2, 'w') as file:
    for model in algorithms:
        averages = calculate_averages(model)
        for k in [10, 20, 50]:
            precision = averages[k]['Precision']
            recall = averages[k]['Recall']
            f1 = averages[k]['F1']
            file.write(f"Average Metrics for {model} @ {k}: {{'Precision': {precision:.4f}, 'Recall': {recall:.4f}, 'F1': {f1:.4f}}}\n")
            print(f"Average Metrics for {model} @ {k}: {{'Precision': {precision:.4f}, 'Recall': {recall:.4f}, 'F1': {f1:.4f}}}")

