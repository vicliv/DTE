"""
File from ADBench https://github.com/Minqi824/ADBench.git 
that was modified to integrate the new semi-supervised setting
and add more flexibility.
"""

import numpy as np
import pandas as pd
import random
import os
from math import ceil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from adbench.myutils import Utils
import pkg_resources

# Path to adbench datasets
DATA_PATH = pkg_resources.resource_filename('adbench', 'datasets/')

# currently, data generator only supports for generating the binary classification datasets
class DataGenerator():
    def __init__(self, seed:int=42, dataset:str=None, test_size:float=0.3,
                 generate_duplicates=True, n_samples_threshold=1000, normal=False):
        '''
        :param seed: seed for reproducible results
        :param dataset: specific the dataset name
        :param test_size: testing set size
        :param generate_duplicates: whether to generate duplicated samples when sample size is too small
        :param n_samples_threshold: threshold for generating the above duplicates, if generate_duplicates is False, then datasets with sample size smaller than n_samples_threshold will be dropped
        '''

        self.seed = seed
        self.dataset = dataset
        self.test_size = test_size

        self.generate_duplicates = generate_duplicates
        self.n_samples_threshold = n_samples_threshold
        self.normal = normal

        # dataset list
        self.dataset_list_classical = [os.path.splitext(_)[0] for _ in os.listdir(DATA_PATH + 'Classical')
                                       if os.path.splitext(_)[1] == '.npz'] # classical AD datasets
        self.dataset_list_cv = [os.path.splitext(_)[0] for _ in os.listdir(DATA_PATH + 'CV_by_ResNet18')
                                if os.path.splitext(_)[1] == '.npz'] # CV datasets
        self.dataset_list_nlp = [os.path.splitext(_)[0] for _ in os.listdir(DATA_PATH + 'NLP_by_BERT')
                                 if os.path.splitext(_)[1] == '.npz'] # NLP datasets

        # myutils function
        self.utils = Utils()

    '''
    Here we also consider the robustness of baseline models, where three types of noise can be added
    1. Duplicated anomalies, which should be added to training and testing set, respectively
    2. Irrelevant features, which should be added to both training and testing set
    3. Annotation errors (Label flips), which should be only added to the training set
    '''
    def add_duplicated_anomalies(self, X, y, duplicate_times:int):
        if duplicate_times <= 1:
            pass
        else:
            # index of normal and anomaly data
            idx_n = np.where(y==0)[0]
            idx_a = np.where(y==1)[0]

            # generate duplicated anomalies
            idx_a = np.random.choice(idx_a, int(len(idx_a) * duplicate_times))

            idx = np.append(idx_n, idx_a); random.shuffle(idx)
            X = X[idx]; y = y[idx]

        return X, y

   
    def generator(self, X=None, y=None, scale=True,
                  la=None, at_least_one_labeled=False,
                  noise_type=None, duplicate_times:int=2, max_size=10000):
        '''
        la: labeled anomalies, can be either the ratio of labeled anomalies or the number of labeled anomalies
        at_least_one_labeled: whether to guarantee at least one labeled anomalies in the training set
        '''

        # set seed for reproducible results
        self.utils.set_seed(self.seed)

        # load dataset
        if self.dataset is None:
            assert X is not None and y is not None, "For customized dataset, you should provide the X and y!"
        else:
            if self.dataset in self.dataset_list_classical:
                data = np.load(os.path.join(DATA_PATH, 'Classical', self.dataset + '.npz'), allow_pickle=True)
            elif self.dataset in self.dataset_list_cv:
                data = np.load(os.path.join(DATA_PATH, 'CV_by_ResNet18', self.dataset + '.npz'), allow_pickle=True)
            elif self.dataset in self.dataset_list_nlp:
                data = np.load(os.path.join(DATA_PATH, 'NLP_by_BERT', self.dataset + '.npz'), allow_pickle=True)
            else:
                raise NotImplementedError

            X = data['X']
            y = data['y']
        
        # number of labeled anomalies in the original data
        if type(la) == float:
            if at_least_one_labeled:
                n_labeled_anomalies = ceil(sum(y) * (1 - self.test_size) * la)
            else:
                n_labeled_anomalies = int(sum(y) * (1 - self.test_size) * la)
        elif type(la) == int:
            n_labeled_anomalies = la
        else:
            raise NotImplementedError

        # if the dataset is too small, generating duplicate smaples up to n_samples_threshold
        if len(y) < self.n_samples_threshold and self.generate_duplicates:
            print(f'generating duplicate samples for dataset {self.dataset}...')
            self.utils.set_seed(self.seed)
            idx_duplicate = np.random.choice(np.arange(len(y)), self.n_samples_threshold, replace=True)
            X = X[idx_duplicate]
            y = y[idx_duplicate]

        # if the dataset is too large, subsampling for considering the computational cost
        if len(y) > max_size:
            print(f'subsampling for dataset {self.dataset}...')
            self.utils.set_seed(self.seed)
            idx_sample = np.random.choice(np.arange(len(y)), max_size, replace=False)
            X = X[idx_sample]
            y = y[idx_sample]

        # show the statistic
        self.utils.data_description(X=X, y=y)

        # spliting the current data to the training set and testing set
        if not self.normal:
            if self.test_size == 0:
                X_train = X.copy()
                y_train = y.copy()
                X_test = X.copy()
                y_test = y.copy()
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, shuffle=True, stratify=y)
        else:
            indices = np.arange(len(X))
            normal_indices = indices[y == 0]
            anomaly_indices = indices[y == 1]

            train_size = round((1-self.test_size) * normal_indices.size)
            train_indices, test_indices = normal_indices[:train_size], normal_indices[train_size:]
            test_indices = np.append(test_indices, anomaly_indices)
        
            X_train = X[train_indices]
            y_train = y[train_indices]
            X_test = X[test_indices]
            y_test = y[test_indices]

        # standard scaling
        if scale:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            
            col_mean = np.nanmean(X_train, axis=0)
            inds = np.where(np.isnan(col_mean))
            col_mean[inds] = 0

            inds = np.where(np.isnan(X_train))
            X_train[inds] = np.take(col_mean, inds[1])
            
            col_mean = np.nanmean(X_test, axis=0)
            inds = np.where(np.isnan(col_mean))
            col_mean[inds] = 0

            inds = np.where(np.isnan(X_test))
            X_test[inds] = np.take(col_mean, inds[1])

        # idx of normal samples and unlabeled/labeled anomalies
        idx_normal = np.where(y_train == 0)[0]
        idx_anomaly = np.where(y_train == 1)[0]

        if type(la) == float:
            if at_least_one_labeled:
                idx_labeled_anomaly = np.random.choice(idx_anomaly, ceil(la * len(idx_anomaly)), replace=False)
            else:
                idx_labeled_anomaly = np.random.choice(idx_anomaly, int(la * len(idx_anomaly)), replace=False)
        elif type(la) == int:
            if la > len(idx_anomaly):
                raise AssertionError(f'the number of labeled anomalies are greater than the total anomalies: {len(idx_anomaly)} !')
            else:
                idx_labeled_anomaly = np.random.choice(idx_anomaly, la, replace=False)
        else:
            raise NotImplementedError

        idx_unlabeled_anomaly = np.setdiff1d(idx_anomaly, idx_labeled_anomaly)
        # whether to remove the anomaly contamination in the unlabeled data
        if noise_type == 'anomaly_contamination':
            idx_unlabeled_anomaly = self.remove_anomaly_contamination(idx_unlabeled_anomaly, contam_ratio)

        # unlabel data = normal data + unlabeled anomalies (which is considered as contamination)
        idx_unlabeled = np.append(idx_normal, idx_unlabeled_anomaly)

        del idx_anomaly, idx_unlabeled_anomaly

        # the label of unlabeled data is 0, and that of labeled anomalies is 1
        y_train[idx_unlabeled] = 0
        y_train[idx_labeled_anomaly] = 1

        return {'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test}