import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch_geometric.nn import BatchNorm,GraphNorm
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score
import numpy as np

def get_device():
    n_gpus = 1
    if torch.cuda.is_available():
        device = 'cuda' 
        n_gpus = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device,n_gpus

def get_normalization(norm_type) -> Module:
    if norm_type=="graph_norm":
        return GraphNorm
    elif norm_type=="batch_norm":
        return BatchNorm
    else:
        raise NameError("invalid norm_type name")
    
def get_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")
    
def evaluate_graph_embeddings_using_svm(embeddings, labels):
    labels = labels.reshape((labels.shape[0]),)
    auc = []
    ap_scores = []
    accuracies = []
    micro_f1s = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    
    for train_index, val_index in kf.split(embeddings, labels):
        x_train,x_test = embeddings[train_index],embeddings[val_index]
        y_train,y_test = labels[train_index], labels[val_index]
        params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}
        svc = SVC(random_state=42,probability=True)
        clf = GridSearchCV(svc, params)
        clf.fit(x_train, y_train)
        proba = clf.predict_proba(x_test)[:, 1]
        preds = clf.predict(x_test)
        
        auc.append(roc_auc_score(y_test,proba))
        ap_scores.append(average_precision_score(y_test,proba))
        accuracies.append(accuracy_score(y_test, preds))
        micro_f1s.append(f1_score(y_test, preds, average="micro"))
    
    metrics = {
        "auc": round(np.mean(auc),3),
        "avg_precision": round(np.mean(ap_scores),3),
        "accuracy":round(np.mean(accuracies),3),
        "micro_f1":round(np.mean(micro_f1s),3)
    }

    return metrics