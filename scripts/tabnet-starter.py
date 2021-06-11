#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer

train = pd.read_csv("../data/train.csv")

# # Pre-Processing 👎🏻 -> 👍

# Normalization
for i in range(75):
    mean, std = train[f'feature_{i}'].mean(), train[f'feature_{i}'].std()
    train[f'feature_{i}'] = train[f'feature_{i}'].apply(lambda x : (x-mean)/std)

# Train, Test, Validation Split
target = 'target'
if "Set" not in train.columns:
    train["Set"] = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(train.shape[0],))

train_indices = train[train.Set=="train"].index
valid_indices = train[train.Set=="valid"].index
test_indices = train[train.Set=="test"].index

nunique = train.nunique()
types = train.dtypes

categorical_columns = []
categorical_dims =  {}
for col in train.columns:
    if types[col] == 'object':
        print(col, train[col].nunique())
        l_enc = LabelEncoder()
        train[col] = l_enc.fit_transform(train[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)
    else:
        train.fillna(train.loc[train_indices, col].mean(), inplace=True)


# # The Model 👷‍♀️


# Columns not to use
unused_feat = ['Set']

# Features to Use
features = [ col for col in train.columns if col not in unused_feat+[target]] 

X_train = train[features].values[train_indices]
y_train = train[target].values[train_indices]

X_valid = train[features].values[valid_indices]
y_valid = train[target].values[valid_indices]

X_test = train[features].values[test_indices]
y_test = train[target].values[test_indices]


# Basic model parameters
max_epochs = 30
batch_size = 1024
opt = torch.optim.Adam # Optimizer
opt_params = dict(lr=1e-3)
sch = torch.optim.lr_scheduler.StepLR # LR Scheduler
sch_params = {"step_size":10, "gamma":0.9}
mask = 'entmax'
workers = 2 # For torch DataLoader
sample_type = 1 # For automated sampling with inverse class occurrences 
virtual_batch = 128 # Size of the mini batches used for "Ghost Batch Normalization"


unsupervised_model = TabNetPretrainer(
    optimizer_fn = opt,
    optimizer_params = opt_params,
    mask_type = mask)

clf = TabNetClassifier(gamma = 1.5,
                    lambda_sparse = 1e-4,
                    optimizer_fn = opt,
                    optimizer_params = opt_params,
                    scheduler_fn = sch,
                    scheduler_params = sch_params,
                    mask_type = mask)

# # Training 💪🏻

unsupervised_model.fit(
    X_train=X_train,
    eval_set=[X_valid],
    pretraining_ratio=0.8)

clf.fit(X_train=X_train, 
    y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'val'],
    eval_metric=["logloss", 'balanced_accuracy'],
    max_epochs=max_epochs , patience=15,
    batch_size=batch_size,
    virtual_batch_size=virtual_batch,
    num_workers=workers,
    weights=sample_type,
    drop_last=False,
    from_unsupervised=unsupervised_model)

# # Submission

test = pd.read_csv('../data/test.csv')
test_indices = test.index
test_ds = test[features].values[test_indices]

sample_submission = pd.read_csv('../data/sample_submission.csv')
sample_submission[['Class_1','Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']] = clf.predict_proba(test_ds)
sample_submission.to_csv('tabnet_submission.csv',index = False)
