import numpy as np
import xarray as xr
import pandas as pd
import copy
from datetime import datetime, timedelta
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import make_scorer
from sklearn.metrics import log_loss
from sklearn.utils.class_weight import compute_class_weight
import sys
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import glob 


from sklearn import datasets, ensemble
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split

from sklearn.utils import class_weight
import json

import xgboost as xgb
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import shap
from sklearn.model_selection import StratifiedKFold, KFold

import argparse

parser = argparse.ArgumentParser(description="Process command line arguments.")

# Adding arguments
parser.add_argument('gpu_id', type=int, help='The GPU ID to use')
parser.add_argument('ivar_start', type=int, help='The starting value for ivar')
parser.add_argument('ivar_end', type=int, help='The ending value for ivar')
parser.add_argument('week', type=int, help='The week of forecast')

# Parsing arguments
args = parser.parse_args()

gpu_id = args.gpu_id
ivar_start = args.ivar_start
ivar_end = args.ivar_end
week_out = args.week

print(gpu_id,ivar_start,ivar_end,week_out)

import os

# Run nvidia-smi to get GPU information
os.system('nvidia-smi')

path_wrs = '/glade/work/jhayron/Data4Predictability/WR_Series_v20241226.csv'
path_weekly_anoms = '/glade/derecho/scratch/jhayron/Data4Predictability/WeeklyAnoms_DetrendedStd_v3_2dg/'

# ### LOAD DATA WRs #####

wr_original_series = pd.read_csv(path_wrs,\
                index_col=0,names=['week0','dist'],skiprows=1,parse_dates=True)

### Do it only with the cold season ###
wr_original_series.loc[(wr_original_series.index.month<=9)&(wr_original_series.index.month>=4),'week0']=np.nan

# Rolling window for mode
rolling_mode = (
    wr_original_series.rolling('7d', center=True,min_periods=7)
    .apply(lambda x: x.mode()[0] if not x.mode().empty else float('nan'))
).shift(-3)

# Rolling window for the count of the mode
rolling_mode_count = (
    wr_original_series.rolling('7d', center=True,min_periods=7)
    .apply(lambda x: (x == x.mode()[0]).sum() if not x.mode().empty else 0)
).shift(-3)

# If duration of WR during week was less than 4, assing NO WR class
rolling_mode.loc[rolling_mode_count['week0']<4,'week0'] = 4
wr_series_mode = copy.deepcopy(rolling_mode)
time_index = pd.to_datetime(wr_series_mode.index).dayofweek
wr_series_mode = wr_series_mode.iloc[time_index.isin([0,3])].dropna()
wr_series = copy.deepcopy(wr_series_mode)

for wk in range(2,10):
    series_temp = copy.deepcopy(wr_series["week0"])
    series_temp.index = series_temp.index - timedelta(weeks = wk-1)
    series_temp.name = f'week{wk-1}'
    if wk==2:
        df_shifts = pd.concat([pd.DataFrame(wr_series["week0"]),pd.DataFrame(series_temp)],axis=1)  
    else:
        df_shifts = pd.concat([df_shifts,pd.DataFrame(series_temp)],axis=1)

list_files_anoms = np.sort(glob.glob(f'{path_weekly_anoms}*.nc'))
list_vars = [list_files_anoms[i].split('/')[-1][:-3] for i in range(len(list_files_anoms))]

# Define a boxcar filter function
def boxcar_filter(data, size):
    kernel = np.ones((size, size)) / (size * size)
    from scipy.signal import convolve2d
    return convolve2d(data, kernel, mode="same", boundary="fill", fillvalue=np.nan)

for ivar in range(ivar_start,ivar_end+1):
    print('*****************************************************************************')
    print('*****************************************************************************')
    print(ivar,list_vars[ivar])
    path_nc_anoms = f'{path_weekly_anoms}{list_vars[ivar]}.nc'
    anoms = xr.open_dataset(path_nc_anoms)
    anoms = anoms.assign_coords(time=pd.DatetimeIndex(anoms.time).normalize())
    var_name_nc = list(anoms.data_vars.keys())[0]
    
    # Apply boxcar filter
    smoothed_anoms = xr.apply_ufunc(
        boxcar_filter,
        anoms,
        kwargs={"size": 3},  # Adjust window size (e.g., 5x5 grid cells)
        input_core_dims=[["lat", "lon"]],
        output_core_dims=[["lat", "lon"]],
        vectorize=True,
    )
    
    anoms_flattened = smoothed_anoms[var_name_nc].stack(flat_spatial=('lat', 'lon'))
    anoms_flattened_og = copy.deepcopy(anoms_flattened)
    anoms_flattened = pd.DataFrame(anoms_flattened,index = anoms_flattened.time)
    anoms_flattened = anoms_flattened.dropna(axis=1, how='any')
    
    combined_df = copy.deepcopy(anoms_flattened)
    
    full_df = copy.deepcopy(combined_df)
    full_df['day_sin'] = np.sin(2 * np.pi * full_df.index.day_of_year / 365)
    full_df['day_cos'] = np.cos(2 * np.pi * full_df.index.day_of_year / 365)
    
    start_time = datetime.now()
    print(f'WEEK: {week_out}')
    week_out_str = f'week{week_out}'
    
    fully_combined_df = pd.concat([full_df,df_shifts[week_out_str]],axis=1)
    fully_combined_df = fully_combined_df.dropna()
    
    # Step 1: Split the data into training+validation and testing sets
    trainval_df, test_df = train_test_split(fully_combined_df, test_size=0.25, shuffle=False)
    
    kf = KFold(n_splits=3, shuffle=False)
    
    X_trainval = trainval_df.iloc[:,:-1]
    y_trainval = trainval_df.iloc[:,-1]
    
    X_test = test_df.iloc[:,:-1]
    y_test = test_df.iloc[:,-1]
    
    folds = []
    for train_index, val_index in kf.split(X_trainval, y_trainval):
        folds.append((train_index, val_index))
    
    import random
    # Set the seed for reproducibility
    seed_value = 0
    random.seed(seed_value)
    np.random.seed(seed_value)
    # Function to randomly sample from the bounds
    def random_sample_hyperparams(pbounds, n_samples):
        sampled_params = []
        for _ in range(n_samples):
            params = {}
            for key, (low, high) in pbounds.items():
                if 'log10' in key:  # Log-scale sampling
                    params[key] = 10 ** random.uniform(low, high)
                elif isinstance(low, int) and isinstance(high, int):  # Integer sampling
                    params[key] = random.randint(low, high)
                else:  # Continuous sampling
                    params[key] = random.uniform(low, high)
            sampled_params.append(params)
        return sampled_params
    
    def brier_multi(targets, probs):
        return np.mean(np.sum((probs - targets)**2, axis=1))
    
    # Define the bounds for the hyperparameters
    pbounds = {
        'max_depth': (2, 30),  # Tree depth
        'min_child_weight': (1, 20),  # Minimum sum of instance weights in a leaf
        'subsample': (0.7, 0.9),  # Subsample ratio of training data
        'colsample_bytree': (0.05, 1),  # Feature sampling per tree
        'colsample_bylevel': (0.05, 1),  # Feature sampling per level
        
        'log10_learning_rate': (-3, -0.3),  # Log10 space for learning rate
        'gamma': (0, 20.),  # Regularization term
        'log10_reg_lambda': (0, 2.5),  # L2 regularization (log-scale)
        'log10_reg_alpha': (0, 1.6),  # L1 regularization (log-scale)
        'log10_n_trees': (1., 2.5),  # L1 regularization (log-scale)
        'beta_class_weights': (0., 5.),  # 
    }
    
    # Example: Generate 10 random hyperparameter combinations
    n_samples = 300
    random_params = random_sample_hyperparams(pbounds, n_samples)
    
    import datetime as dt
    print(dt.datetime.now())
    
    scores_full = []
    for ifold in range(len(folds)):
        X_train_fold = X_trainval.iloc[folds[ifold][0]]
        y_train_fold = y_trainval.iloc[folds[ifold][0]]
    
        X_test_fold = X_trainval.iloc[folds[ifold][1]]
        y_test_fold = y_trainval.iloc[folds[ifold][1]]
    
        dtrain_fold = xgb.DMatrix(X_train_fold.values, y_train_fold.values)
        dtest_fold = xgb.DMatrix(X_test_fold.values, y_test_fold.values)
        
        scores_fold = []
        for iparams,params_temp in enumerate(random_params):
            beta_class_weights = params_temp['beta_class_weights']
            class_weights_arr = compute_class_weight('balanced', 
                                                     classes=np.unique(y_train_fold), y=y_train_fold)
            class_weight_dict = dict(zip(np.unique(y_train_fold), class_weights_arr))
            train_weight = np.array([class_weight_dict[label] for label in y_train_fold])**beta_class_weights
            dtrain_fold.set_weight(train_weight)
            
            clf = xgb.XGBClassifier(
                max_depth = params_temp['max_depth'],               # Limit tree depth to avoid overfitting.
                learning_rate = params_temp['log10_learning_rate'],        # Smaller learning rate for stable learning.
                subsample = params_temp['subsample'],             # Use 80% of the training data for each boosting round.
                colsample_bytree = params_temp['colsample_bytree'],      # Use 10% of features for each tree to handle high dimensionality.
                colsample_bylevel = params_temp['colsample_bylevel'],     # Use 10% of features at each level (to reduce computation and avoid overfitting).
                gamma = params_temp['gamma'],                      # Penalize overly complex trees.
                min_child_weight = params_temp['min_child_weight'],# Minimum sum of instance weights in a leaf to prevent small splits.
                reg_alpha = params_temp['log10_reg_alpha'],              # L1 regularization for sparsity in feature selection.
                reg_lambda = params_temp['log10_reg_lambda'],            # L2 regularization to control large weights and prevent overfitting.
                num_class=5,
                objective = "multi:softprob",
                tree_method='hist',
                device = f'cuda:{gpu_id}')
            dic_params_cv = clf.get_xgb_params()
            ### Train model ###
            clf = xgb.train(
                            dic_params_cv,
                            dtrain_fold,
                            num_boost_round=int(params_temp['log10_n_trees'])  # Use the best boosting rounds
                            )
            ### Evaluate ###
            preds = clf.predict(dtest_fold)
            targets = y_test_fold
            y_one_hot = np.eye(5)[targets.values.astype(int)]
            f1_score_micro = f1_score(targets, np.argmax(preds, axis=1), average='micro')  # Use your chosen metric
            f1_score_macro = f1_score(targets, np.argmax(preds, axis=1), average='macro')  # Use your chosen metric
            f1_score_weighted = f1_score(targets, np.argmax(preds, axis=1), average='weighted')  # Use your chosen metric
            brier_score = brier_multi(y_one_hot, preds)
    
            scores_temp = [f1_score_micro,f1_score_macro,f1_score_weighted,brier_score]
            scores_fold.append(scores_temp)
        scores_full.append(scores_fold)
    
    print(dt.datetime.now())
    
    scores_full = np.array(scores_full)
    scores_average_folds = scores_full.mean(axis=0)
    # Change sign of Brier score
    scores_average_folds[:,-1] = -scores_average_folds[:,-1]

    from sklearn.preprocessing import MinMaxScaler
    
    # Normalize the scores
    scaler = MinMaxScaler()
    normalized_scores = scaler.fit_transform(scores_average_folds)
    
    def is_dominated(candidate, others):
        """
        Checks if a candidate is dominated by any other combination in the list.
        """
        for other in others:
            if all(o >= c for c, o in zip(candidate, other)) and any(o > c for c, o in zip(candidate, other)):
                return True
        return False
    
    # Find the indices of the Pareto frontier
    pareto_indices = []
    for i, score in enumerate(normalized_scores):
        # Remove the current score and check if it's dominated by others
        others = np.delete(normalized_scores, i, axis=0)
        if not is_dominated(score, others):
            pareto_indices.append(i)
    
    # Get the Pareto frontier combinations and their original scores
    pareto_frontier = normalized_scores[pareto_indices]
    pareto_original_scores = scores_average_folds[pareto_indices]
    
    avg_norm_scores = []
    for p_index in pareto_indices:
        avg_norm_scores.append(normalized_scores[p_index].mean())
    best_params = random_params[pareto_indices[np.argmax(avg_norm_scores)]]
    
    clf = xgb.XGBClassifier(
        max_depth = best_params['max_depth'],               # Limit tree depth to avoid overfitting.
        learning_rate = best_params['log10_learning_rate'],        # Smaller learning rate for stable learning.
        subsample = best_params['subsample'],             # Use 80% of the training data for each boosting round.
        colsample_bytree = best_params['colsample_bytree'],      # Use 10% of features for each tree to handle high dimensionality.
        colsample_bylevel = best_params['colsample_bylevel'],     # Use 10% of features at each level (to reduce computation and avoid overfitting).
        gamma = best_params['gamma'],                      # Penalize overly complex trees.
        min_child_weight = best_params['min_child_weight'],# Minimum sum of instance weights in a leaf to prevent small splits.
        reg_alpha = best_params['log10_reg_alpha'],              # L1 regularization for sparsity in feature selection.
        reg_lambda = best_params['log10_reg_lambda'],            # L2 regularization to control large weights and prevent overfitting.
        num_class=5,
        objective = "multi:softprob",
        tree_method='hist',
        device = f'cuda:{gpu_id}')
    dic_params_cv = clf.get_xgb_params()
    
    ### Train model ###
    beta_class_weights = best_params['beta_class_weights']
    class_weights_arr = compute_class_weight('balanced', 
                                             classes=np.unique(y_trainval), y=y_trainval)
    class_weight_dict = dict(zip(np.unique(y_trainval), class_weights_arr))
    train_weight = np.array([class_weight_dict[label] for label in y_trainval])**beta_class_weights
    
    dtrain = xgb.DMatrix(X_trainval.values, y_trainval.values, weight=train_weight)
    dtest = xgb.DMatrix(X_test.values, y_test.values)
    clf = xgb.train(
                    dic_params_cv,
                    dtrain,
                    num_boost_round=int(best_params['log10_n_trees'])  
                    )
    
    ### Evaluate ###
    preds = clf.predict(dtest)
    targets = y_test
    y_one_hot = np.eye(5)[targets.values.astype(int)]
    f1_score_micro = f1_score(targets, np.argmax(preds, axis=1), average='micro')  # Use your chosen metric
    f1_score_macro = f1_score(targets, np.argmax(preds, axis=1), average='macro')  # Use your chosen metric
    f1_score_weighted = f1_score(targets, np.argmax(preds, axis=1), average='weighted')  # Use your chosen metric
    brier_score = brier_multi(y_one_hot, preds)
    
    print(f1_score_micro,f1_score_macro,f1_score_weighted,brier_score)
    print(best_params)
    print(np.bincount(np.argmax(preds, axis=1)))
    print(np.bincount(targets))
    
    ## Collect shap from correct predictions in validation set
    import shap
    shap_correct_pos = []
    shap_correct_neg = []
    for ifold in range(len(folds)):
        X_train_fold = X_trainval.iloc[folds[ifold][0]]
        y_train_fold = y_trainval.iloc[folds[ifold][0]]
    
        X_test_fold = X_trainval.iloc[folds[ifold][1]]
        y_test_fold = y_trainval.iloc[folds[ifold][1]]
    
        beta_class_weights = best_params['beta_class_weights']
        class_weights_arr = compute_class_weight('balanced', 
                                                 classes=np.unique(y_train_fold), y=y_train_fold)
        class_weight_dict = dict(zip(np.unique(y_train_fold), class_weights_arr))
        train_weight = np.array([class_weight_dict[label] for label in y_train_fold])**beta_class_weights
    
        dtrain_fold = xgb.DMatrix(X_train_fold.values, y_train_fold.values, weight=train_weight)
        dtest_fold = xgb.DMatrix(X_test_fold.values, y_test_fold.values)
    
        ## Train model
        
        clf = xgb.XGBClassifier(
            max_depth = best_params['max_depth'],               # Limit tree depth to avoid overfitting.
            learning_rate = best_params['log10_learning_rate'],        # Smaller learning rate for stable learning.
            subsample = best_params['subsample'],             # Use 80% of the training data for each boosting round.
            colsample_bytree = best_params['colsample_bytree'],      # Use 10% of features for each tree to handle high dimensionality.
            colsample_bylevel = best_params['colsample_bylevel'],     # Use 10% of features at each level (to reduce computation and avoid overfitting).
            gamma = best_params['gamma'],                      # Penalize overly complex trees.
            min_child_weight = best_params['min_child_weight'],# Minimum sum of instance weights in a leaf to prevent small splits.
            reg_alpha = best_params['log10_reg_alpha'],              # L1 regularization for sparsity in feature selection.
            reg_lambda = best_params['log10_reg_lambda'],            # L2 regularization to control large weights and prevent overfitting.
            num_class=5,
            objective = "multi:softprob",
            tree_method='hist',
            device = f'cuda:{gpu_id}')
        dic_params_cv = clf.get_xgb_params()
        ### Train model ###
        clf = xgb.train(
                        dic_params_cv,
                        dtrain_fold,
                        num_boost_round=int(best_params['log10_n_trees'])  # Use the best boosting rounds
                            )
        ### Evaluate ###
        preds = clf.predict(dtest_fold)
        targets = y_test_fold
        y_one_hot = np.eye(5)[targets.values.astype(int)]
        f1_score_micro = f1_score(targets, np.argmax(preds, axis=1), average='micro')  # Use your chosen metric
        f1_score_macro = f1_score(targets, np.argmax(preds, axis=1), average='macro')  # Use your chosen metric
        f1_score_weighted = f1_score(targets, np.argmax(preds, axis=1), average='weighted')  # Use your chosen metric
        brier_score = brier_multi(y_one_hot, preds)
        
        scores_temp = [f1_score_micro,f1_score_macro,f1_score_weighted,brier_score]
        ### SHAP ###
        explainer = shap.Explainer(clf)
        shap_values = explainer(X_test_fold.values)
        where_correct = np.where(np.argmax(preds, axis=1) == y_test_fold.values)[0]
        for itime_correct in where_correct:
            shap_temp = shap_values.values[itime_correct]
            
            shap_temp_4pos = shap_temp[:,int(y_test_fold.iloc[itime_correct])]
            shap_temp_4pos[shap_temp_4pos<0] = 0
            
            shap_temp_4neg = np.delete(shap_temp,int(y_test_fold.iloc[itime_correct]),axis=1)
            shap_temp_4neg[shap_temp_4neg>0] = 0
            
            shap_correct_pos.append(shap_temp_4pos)
            shap_correct_neg.append(shap_temp_4neg.mean(axis=1))
            
    shap_correct_pos = np.array(shap_correct_pos)
    shap_correct_neg = np.array(shap_correct_neg)
    
    mean_shap_correct_pos = abs(shap_correct_pos).mean(axis=0)
    mean_shap_correct_neg = abs(shap_correct_neg).mean(axis=0)
    
    np.save(f'FilesIndTests/{list_vars[ivar]}_{week_out_str}_best_params.npy',best_params)
    np.save(f'FilesIndTests/{list_vars[ivar]}_{week_out_str}_scores_full.npy',scores_full)
    np.save(f'FilesIndTests/{list_vars[ivar]}_{week_out_str}_random_params.npy',random_params)
    np.save(f'FilesIndTests/{list_vars[ivar]}_{week_out_str}_shap_correct_pos.npy',shap_correct_pos)
    np.save(f'FilesIndTests/{list_vars[ivar]}_{week_out_str}_shap_correct_neg.npy',shap_correct_neg)