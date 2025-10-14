import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

class BaselineModels:
    def __init__(self):
        self.ocsvm = OneClassSVM(gamma='auto')
        self.iforest = IsolationForest(random_state=42)

    def fit_ocsvm(self, X_train):
        self.ocsvm.fit(X_train)

    def predict_ocsvm(self, X_test):
        # OneClassSVM: -1 = anomaly, 1 = normal
        return (self.ocsvm.predict(X_test) == -1).astype(int)

    def fit_iforest(self, X_train):
        self.iforest.fit(X_train)

    def predict_iforest(self, X_test):
        # IsolationForest: -1 = anomaly, 1 = normal
        return (self.iforest.predict(X_test) == -1).astype(int)

    def evaluate(self, y_true, y_pred):
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
