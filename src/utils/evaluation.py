from collections import defaultdict
import numpy as np
import pandas as pd
from ..recommenders.collaborative import train_svd_model

ratings = pd.read_csv('data/raw/dataset1/ratings.csv', usecols=['userId', 'movieId', 'rating'])

svd_model, predictions, rmse, testset = train_svd_model(ratings)

def precision_recall_at_k(predictions=predictions, k=10, threshold=4.0):    
    user_recs = defaultdict(list)
    user_truth = defaultdict(list)

    for pred in predictions:
        uid = pred.uid
        user_recs[uid].append((pred.iid, pred.est))
        if pred.r_ui >= threshold:
            user_truth[uid].append(pred.iid)

    precisions, recalls = [], []

    for uid in user_recs:
        user_recs[uid].sort(key=lambda x: x[1], reverse=True)
        top_k = [iid for iid, _ in user_recs[uid][:k]]
        relevant = set(user_truth[uid])
        recommended = set(top_k)
        true_positives = relevant & recommended

        if recommended:
            precisions.append(len(true_positives) / len(recommended))
        if relevant:
            recalls.append(len(true_positives) / len(relevant))

    return np.mean(precisions), np.mean(recalls)

# Use predictions on test_data
predictions = svd_model.test(testset)

p, r = precision_recall_at_k(predictions, k=10)
print(f"Precision@10: {p:.4f}  Recall@10: {r:.4f}")
