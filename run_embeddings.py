from adbench.baseline.PyOD import PYOD
from baselines.dagmm import DAGMM
from baselines.drocc import DROCC
from baselines.normalizing_flow import FlowModel
from baselines.goad import GOAD
from baselines.icl import ICL

import argparse
import numpy as np

from diffusion.dte import DTECategorical, DTEInverseGamma
from diffusion.non_param_dte import DTENonParametric
from diffusion.ddpm import DDPM

import os
import pandas as pd

import time

from adbench.myutils import Utils

import sklearn.metrics as skm
from data_generator import DataGenerator

def low_density_anomalies(test_log_probs, num_anomalies):
    """ Helper function for the F1-score, selects the num_anomalies lowest values of test_log_prob
    """
    anomaly_indices = np.argpartition(test_log_probs, num_anomalies-1)[:num_anomalies]
    preds = np.zeros(len(test_log_probs))
    preds[anomaly_indices] = 1
    return preds

def main(args):
    seed = args.seed
    setting = args.setting
    
    visa_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

    
    
    
    utils = Utils() # utils function
    utils.set_seed(seed)
    
    # Get the datasets from ADBench
    dataset_resnet34 = [os.path.splitext(_)[0] for _ in os.listdir("vision/data/resnet34")
                                if os.path.splitext(_)[1] == '.npz'] # classical AD datasets
    dataset_vicreg = [os.path.splitext(_)[0] for _ in os.listdir("vision/data/vicreg")
                                if os.path.splitext(_)[1] == '.npz'] # classical AD datasets

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
    model_dict['DTE-NP'] = DTENonParametric
    # model_dict['DTE-IG'] = DTEInverseGamma
    model_dict['DTE-C'] = DTECategorical
    model_dict['KNN'] = PYOD

    dir = './results/embeddings/resnet34/'
    
    if not os.path.exists(dir):
        os.makedirs(dir)
    
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
        
    def run_dataset(dataset, model):
        '''
        la: ratio of labeled anomalies, from 0.0 to 1.0
        realistic_synthetic_mode: types of synthetic anomalies, can be local, global, dependency or cluster
        noise_type: inject data noises for testing model robustness, can be duplicated_anomalies, irrelevant_features or label_contamination
        '''
        print(dataset)
        
        if dataset in visa_list:
            datagenerator = DataGenerator(seed = seed, test_size=0.1, normal=True) # data generator
        else:
            datagenerator = DataGenerator(seed = seed, test_size=0.2, normal=True) # data generator
        
        # import the dataset
        data =  np.load("vision/data/" + model + '/' + dataset + '.npz', allow_pickle=True)
        X = data['X']
        y = data['y']
        data = datagenerator.generator(X = X, y = y, la=0, max_size=100000) # maximum of 50,000 data points are available
        
        for name, clf in model_dict.items():
            # model initialization
            clf = clf(seed=seed, model_name=name)
            print(name)
            
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
    
    for dataset in dataset_resnet34:
        run_dataset(dataset, "resnet34")
    
    dir = './results/embeddings/vicreg/'

    if not os.path.exists(dir):
        os.makedirs(dir)
    
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
    for dataset in dataset_vicreg:
        run_dataset(dataset, "vicreg")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--setting', type=str,
        default='semi', help='choice of experimental setting (semi or unsup)')
    parser.add_argument('--seed', type=int, 
        default=42, help='random seed')

    args = parser.parse_args()
    main(args)
