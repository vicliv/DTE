from adbench.baseline.PyOD import PYOD
from baselines.dagmm import DAGMM
from baselines.drocc import DROCC
from baselines.normalizing_flow import FlowModel
from baselines.goad import GOAD
from baselines.icl import ICL


import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import random

import argparse
import numpy as np

from vision.dte_cv import DTECategorical, DTEInverseGamma
from diffusion.non_param_dte import DTENonParametric
from vision.ddpm_cv import DDPM

import os
import pandas as pd
import torch

import time

from adbench.myutils import Utils

import sklearn.metrics as skm
from data_generator import DataGenerator

def get_MNIST(anomaly_class = 0):
        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.0,), (0.25,))])

        dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        
        i = anomaly_class

        normal_data = [data for data in dataset if data[1] != i] # Assuming "airplane" class as "normal"
        anomaly_data = [data for data in dataset if data[1] == i]
        # # Assigning labels
        normal_data = [(x[0], 0) for x in normal_data]
        anomaly_data = [(x[0], 1) for x in anomaly_data]

        # Combine and shuffle
        final_data = normal_data + anomaly_data
        random.shuffle(final_data)
        data = list(zip(*final_data))
        return torch.stack(list(data[0])).numpy(), np.array(list(data[1]))

def get_CIFAR10(anomaly_class = 0):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))])

        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        
        i = anomaly_class

        normal_data = [data for data in dataset if data[1] != i] # Assuming "airplane" class as "normal"
        anomaly_data = [data for data in dataset if data[1] == i]
        # # Assigning labels
        normal_data = [(x[0], 0) for x in normal_data]
        anomaly_data = [(x[0], 1) for x in anomaly_data]

        # Combine and shuffle
        final_data = normal_data + anomaly_data
        random.shuffle(final_data)
        data = list(zip(*final_data))
        return torch.stack(list(data[0])).numpy(), np.array(list(data[1]))

def get_VISA(dataset):
    folder = os.path.join("VisA_pytorch", "1cls")
    
    train_folder = os.path.join(folder, dataset, 'train')
    test_folder = os.path.join(folder, dataset, 'test')
             
    transform = transforms.Compose([transforms.Resize(320, interpolation=transforms.InterpolationMode.BICUBIC),
                                    transforms.CenterCrop(300),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.0, 0.0, 0.0), (0.25, 0.25, 0.25)),
                                    ])

    dataset = ImageFolder(root=train_folder, transform=transform)

    train_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, drop_last=False)

    inputs = []
    labels = []
    with torch.no_grad():
        for i, d in enumerate(train_loader):
            X, y = d

            inputs.append(X)
            labels.append(y)

    dataset = ImageFolder(root=test_folder, transform=transform)

    test_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, drop_last=False)

    with torch.no_grad():
        for i, d in enumerate(test_loader):
            X, y = d

            inputs.append(X)
            labels.append(1-y)

    X = np.vstack(inputs)
    y = np.hstack(labels)
    
    return X, y

def low_density_anomalies(test_log_probs, num_anomalies):
    """ Helper function for the F1-score, selects the num_anomalies lowest values of test_log_prob
    """
    anomaly_indices = np.argpartition(test_log_probs, num_anomalies-1)[:num_anomalies]
    preds = np.zeros(len(test_log_probs))
    preds[anomaly_indices] = 1
    return preds

def main(args):
    seed = args.seed
    
    dir = './results/images/'
    
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    utils = Utils() # utils function
    utils.set_seed(seed)
    
    visa_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    
    # Get the datasets from ADBench
    dataset_list = ["CIFAR10_" + str(i) for i in range(10)]
    dataset_list.extend(["MNIST_" + str(i) for i in range(10)])
    dataset_list.extend(visa_list)

    dataset = []
    model_dict = {}

    # Select models
    # for _ in ['IForest', 'OCSVM', 'COPOD', 'ECOD', 'FeatureBagging', 'HBOS', 'KNN', 'LODA',
    #                   'LOF', 'MCD', 'PCA', 'DeepSVDD']:
    #     model_dict[_] = PYOD
    

    # model_dict['DAGMM'] = DAGMM
    # model_dict['DROCC'] = DROCC
    # model_dict['GOAD'] = GOAD
    # model_dict['ICL'] = ICL
    # model_dict['PlanarFlow'] = FlowModel
    model_dict['DDPM'] = DDPM
    # model_dict['DTPM-NP'] = DTENonParametric
    # model_dict['DTPM-IG'] = DTEInverseGamma
    model_dict['DTPM-C'] = DTECategorical
    model_dict['KNN'] = PYOD
    
    # Create dataframes to save the results
    aucroc_name = dir + str(seed) + "_AUCROC.csv"
    aucpr_name = dir + str(seed) + "_AUCPR.csv"
    f1_name = dir + str(seed) + "_AUCF1.csv"
    train_name = dir + str(seed) + "_TrainTime.csv"
    inference_name = dir + str(seed) + "_InferenceTime.csv"
    
    try:
        df_AUCROC = pd.read_csv(aucroc_name, index_col = 0) 
    except:
        df_AUCROC = pd.DataFrame(data=None)
    try:
        df_AUCPR = pd.read_csv(aucpr_name, index_col = 0)
    except:
        df_AUCPR = pd.DataFrame(data=None)
    try:
        df_F1 = pd.read_csv(f1_name, index_col = 0)
    except:
        df_F1 = pd.DataFrame(data=None)
    try:
        df_train = pd.read_csv(train_name, index_col = 0)
    except:
        df_train = pd.DataFrame(data=None)
    try:
        df_inference = pd.read_csv(inference_name, index_col = 0)
    except:
        df_inference = pd.DataFrame(data=None)

    for dataset in dataset_list:
        print(dataset)
        if "MNIST" in dataset:
            X, y = get_MNIST(int(dataset.split("_")[1]))
            test_size = 0.2
        elif "CIFAR10" in dataset:
            X, y = get_CIFAR10(int(dataset.split("_")[1]))
            test_size = 0.2
        elif dataset in visa_list:
            X, y = get_VISA(dataset)
            test_size = 0.1
        
        data = {}
        
        if X.shape[1] == 1:
            X = X.repeat(3, 1) # extent the channel if the picture is not colorful
        
        indices = np.arange(len(X))
        normal_indices = indices[y == 0]
        anomaly_indices = indices[y == 1]

        train_size = round((1-test_size) * normal_indices.size)
        train_indices, test_indices = normal_indices[:train_size], normal_indices[train_size:]
        test_indices = np.append(test_indices, anomaly_indices)

        data['X_train'] = X[train_indices]
        data['y_train'] = y[train_indices]
        data['X_test'] = X[test_indices]
        data['y_test'] = y[test_indices]
        
        for name, clf in model_dict.items():
            # model initialization
            clf = clf(seed=seed, model_name=name)
            print(name)
            
            if name == "KNN" or name == "DTE-NP":
                data['X_train'] = data['X_train'].reshape((data['X_train'].shape[0], -1))
                data['X_test'] = data['X_test'].reshape((data['X_test'].shape[0], -1))
            
            # training, for unsupervised models the y label will be discarded
            start_time = time.time()
            clf = clf.fit(X_train=data['X_train'], y_train=data['y_train'])
            end_time = time.time(); time_fit = end_time - start_time 
            
            start_time = time.time()
            if name == 'DAGMM':
                score = clf.predict_score(data['X_train'], data['X_test'])
            else:
                score = clf.predict_score(data['X_test'])
            end_time = time.time(); time_inference = end_time - start_time
            
            indices = np.arange(len(data['y_test']))
            p = low_density_anomalies(-score, len(indices[data['y_test']==1]))
            f1_score = skm.f1_score(data['y_test'], p)
            print('F1 score: ' + str(f1_score))

            df_F1.loc[dataset, name] = f1_score
            df_F1.to_csv(f1_name)

            inds = np.where(np.isnan(score))
            score[inds] = 0
            
            result = utils.metric(y_true=data['y_test'], y_score=score)
            print('AUCROC: ' + str(result['aucroc']))
            
            # save results
            df_AUCROC.loc[dataset, name] = result['aucroc']
            df_AUCPR.loc[dataset, name] = result['aucpr']
            
            df_train.loc[dataset, name] = time_fit
            df_inference.loc[dataset, name] = time_inference
            
            df_AUCROC.to_csv(aucroc_name)
            df_AUCPR.to_csv(aucpr_name)
            
            df_train.to_csv(train_name)
            df_train.to_csv(train_name)
            
            df_inference.to_csv(inference_name)
            df_inference.to_csv(inference_name)
            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--seed', type=int, 
        default=42, help='random seed')

    args = parser.parse_args()
    main(args)
