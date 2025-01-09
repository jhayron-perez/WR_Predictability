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
from sklearn.utils import class_weight
import json

import xgboost as xgb
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import shap

import argparse

parser = argparse.ArgumentParser(description="Process command line arguments.")

# Adding arguments
parser.add_argument('gpu_id', type=int, help='The GPU ID to use')
parser.add_argument('component', type=str, help='component to train')

# Parsing arguments
args = parser.parse_args()

gpu_id = args.gpu_id
component = args.component

path_wrs = '/glade/work/jhayron/Data4Predictability/WR_Series_v20241226.csv'
path_weekly_anoms = '/glade/derecho/scratch/jhayron/Data4Predictability/WeeklyAnoms_DetrendedStd_v3_2dg/'

path_results = '/glade/derecho/scratch/jhayron/Data4Predictability/ResultsXGBoost/Results_v20241226/'
path_results_probs = '/glade/derecho/scratch/jhayron/Data4Predictability/ResultsXGBoost/ResultsProbs_v20241226/'
path_shap= '/glade/derecho/scratch/jhayron/Data4Predictability/ResultsXGBoost/SHAP_Results_v20241226/'
path_hyperparams = '/glade/derecho/scratch/jhayron/Data4Predictability/ResultsXGBoost/ResultsHyperparamOptim_v20241226/'

# ############# FUNCTIONS ##############

def get_train_val_test_periods(full_df):
    dic_train_val = {}
    dic_test = {}
    
    start_of_test_periods = np.arange(1981,2021,10)
    end_of_test_periods = start_of_test_periods + 9
    
    for iperiod in range(len(start_of_test_periods)):
        df_test_temp = full_df[str(start_of_test_periods[iperiod]):str(end_of_test_periods[iperiod])]
        df_trainval_temp = full_df.drop(df_test_temp.index)
        
        dic_train_val[start_of_test_periods[iperiod]] = df_trainval_temp
        dic_test[start_of_test_periods[iperiod]] = df_test_temp
    return dic_train_val, dic_test

def generate_random_forecast(df_week_0, seed_value=42):
    # Set the random seed for reproducibility
    np.random.seed(seed_value)
    
    # Step 1: Get unique classes and their frequencies
    values = df_week_0[df_week_0.keys()[0]].value_counts()
    
    # Step 2: Calculate the probabilities for each class
    classes = values.index  # Unique classes
    probabilities = values / values.sum()  # Normalize to get probability distribution
    
    # Step 3: Generate a random forecast based on the probabilities
    random_forecast = np.random.choice(classes, size=len(df_week_0), p=probabilities)
    
    # Step 4: Return the random forecast as a DataFrame or Series
    forecast_df = pd.DataFrame(random_forecast, index=df_week_0.index, columns=['y_predicted'])
    
    return forecast_df

def generate_random_forecast_probabilities(df_week_0, seed_value=42):
    # Set the random seed for reproducibility
    np.random.seed(seed_value)
    # Step 1: Get unique classes and their frequencies
    values = df_week_0[df_week_0.keys()[0]].value_counts()
    
    # Step 2: Calculate the probabilities for each class
    classes = values.index  # Unique classes
    probabilities = values / values.sum()  # Normalize to get probability distribution
    
    # Step 3: Create a probability forecast for each sample
    # Create a 2D array where each row is the same probability distribution
    prob_matrix = np.tile(probabilities.values, (len(df_week_0), 1))
    
    # Step 4: Return the probability matrix as a DataFrame
    forecast_df = pd.DataFrame(prob_matrix, index=df_week_0.index, columns=classes)[np.arange(len(classes))]
    
    return forecast_df

def generate_random_forecast_with_monthly_probabilities(df_week_0, seed_value=42):
    # Set the random seed for reproducibility
    np.random.seed(seed_value)
    
    # Extract the month from the index (assuming the index is a datetime index)
    df_week_0['month'] = df_week_0.index.month
    
    # Prepare an empty list to store the random forecast
    forecasts = []
    
    # Loop through each month
    for month in range(1, 13):  # Loop through months 1 to 12
        # Filter data for the current month
        month_data = df_week_0[df_week_0['month'] == month]
        
        # Step 1: Get unique classes and their frequencies for the current month
        values = month_data[df_week_0.keys()[0]].value_counts()
        
        # Step 2: Calculate the probabilities for each class in the current month
        classes = values.index  # Unique classes
        probabilities = values / values.sum()  # Normalize to get probability distribution
        
        # Step 3: Generate random forecasts for the current month based on the probabilities
        month_forecast = np.random.choice(classes, size=len(month_data), p=probabilities)
        
        # Store the forecast for the current month
        forecasts.append(pd.Series(month_forecast, index=month_data.index))
    
    # Combine all monthly forecasts into one DataFrame
    forecast_df = pd.concat(forecasts)
    forecast_df = forecast_df.sort_index()  # Sort the index to preserve the original order
    forecast_df = pd.DataFrame(forecast_df,columns=['y_predicted'])
    return forecast_df

def generate_probability_forecast_with_monthly_probabilities(df_week_0, seed_value=42):
    # Set the random seed for reproducibility
    np.random.seed(seed_value)
    
    # Extract the month from the index (assuming the index is a datetime index)
    df_week_0['month'] = df_week_0.index.month
    
    # Prepare an empty DataFrame to store the probability forecasts
    all_probabilities = pd.DataFrame(index=df_week_0.index)
    
    # Loop through each month
    for month in range(1, 13):  # Loop through months 1 to 12
        # Filter data for the current month
        month_data = df_week_0[df_week_0['month'] == month]
        
        if month_data.empty:
            continue  # Skip if there's no data for the month
        
        # Step 1: Get unique classes and their frequencies for the current month
        values = month_data[df_week_0.keys()[0]].value_counts()
        
        # Step 2: Calculate the probabilities for each class in the current month
        classes = values.index  # Unique classes
        probabilities = values / values.sum()  # Normalize to get probability distribution
        
        # Step 3: Create a probability matrix for the current month
        prob_matrix = np.tile(probabilities.values, (len(month_data), 1))
        
        # Create a DataFrame for this month's probabilities with appropriate columns
        month_prob_df = pd.DataFrame(prob_matrix, index=month_data.index, columns=classes)
        
        # Append this month's DataFrame to the overall probability DataFrame
        all_probabilities = pd.concat([all_probabilities, month_prob_df])
    
    # Sort the index to match the original order
    all_probabilities = all_probabilities.sort_index()
    
    # Fill missing columns with zeros for months that do not include certain classes
    all_classes = df_week_0[df_week_0.keys()[0]].unique()
    all_probabilities = all_probabilities.reindex(columns=all_classes, fill_value=0).dropna()[np.arange(len(classes))]
    
    return all_probabilities

def optimize_xgboost(X_trainval,y_trainval,path_save = None):
    ## Apply Bayesian optimization to XGBoost parameters

    def crossval_xgboost(max_depth,
                         log10_learning_rate,
                         subsample,
                         colsample_bytree,
                         colsample_bylevel,
                         gamma,
                         min_child_weight,
                         log10_reg_alpha,
                         log10_reg_lambda,
                         beta_class_weights):
        
        max_depth = int(max_depth)
        learning_rate = 10 ** log10_learning_rate
        reg_alpha = 10 ** log10_reg_alpha
        reg_lambda = 10 ** log10_reg_lambda
        
        # Instantiate the XGBoost model
        clf = xgb.XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            gamma=gamma,
            min_child_weight = min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            num_class=5,
            device=f'cuda:{gpu_id}',
            tree_method='hist',
            objective='multi:softprob',
            random_state=42
        )
        
        dic_params_cv = clf.get_xgb_params()
        
        # Custom cross-validation with TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)  # Adjust number of splits as needed
        scores = []
        for train_index, test_index in tscv.split(X_trainval):
            X_train, X_test = X_trainval[train_index], X_trainval[test_index]
            y_train, y_test = y_trainval.iloc[train_index], y_trainval.iloc[test_index]

            class_weights_arr = compute_class_weight('balanced', 
                                                     classes=np.unique(y_train), y=y_train)
            class_weight_dict = dict(zip(np.unique(y_train), class_weights_arr))
            train_weight = np.array([class_weight_dict[label] for label in y_train])**beta_class_weights
            
            dtrain = xgb.DMatrix(X_train, y_train, weight=train_weight)
            dtest = xgb.DMatrix(X_test, y_test)
            
            # Train the model with early stopping
            clf = xgb.train(
                    dic_params_cv,
                    dtrain,
                    num_boost_round=20  # Use the best boosting rounds
                )
            # Predict and evaluate
            preds = clf.predict(dtest)
            ###### WITH F1 SCORE ########
            score = f1_score(y_test, np.argmax(preds, axis=1), average='micro')  # Use your chosen metric
            # Penalize if model over-predicts one class
            if ((np.bincount(np.argmax(preds, axis=1))\
                    /np.sum(np.bincount(np.argmax(preds, axis=1)))).max() > 0.4)|\
                    ((np.bincount(np.argmax(preds, axis=1))\
                    /np.sum(np.bincount(np.argmax(preds, axis=1)))).min() < 0.05):
                score = score * 0.5
            scores.append(score)
        # print(scores)
        # print(np.mean(scores))
        return np.mean(scores)

    pbounds = {
        # Tree-specific hyperparameters
        'max_depth': (2, 20),  # Moderate depth to prevent overfitting
        'min_child_weight': (1, 20),  # Prevent overly small leaves
        'subsample': (0.7, 0.9),  # Balance between under- and over-sampling
        'colsample_bytree': (0.6, 0.9),  # Use a subset of features to reduce variance
        'colsample_bylevel': (0.75, 1),  # Similar to colsample_bytree but at each split
    
        # Learning task-specific hyperparameters
        'log10_learning_rate': (-4, -1),  # Learning rate in log10 space to explore lower values
        'gamma': (0, 5),  # Regularization term to prevent over-complex trees
        'log10_reg_lambda': (0, 2.5),  # L2 regularization
        'log10_reg_alpha': (0.6, 1.6),  # L1 regularization
    
        # General
        'beta_class_weights': (0, 0.8),  # Use class weights if needed for imbalanced data
    }
    
    acq = acquisition.ExpectedImprovement(xi=0.) ## CHOSEN ONE xi->0 full exploitation
    optimizer = BayesianOptimization(
        f=crossval_xgboost,
        pbounds=pbounds,
        random_state=42,
        verbose=1,
        acquisition_function=acq)
    
    optimizer.maximize(
        init_points=60,
        n_iter=15,
    )
    
    best_params = optimizer.max['params']
    if path_save:
        results_df = pd.DataFrame(optimizer.res)
        params_df = pd.json_normalize(results_df['params'])
        final_df = pd.concat([params_df, results_df['target']], axis=1)
        final_df.to_csv(path_save)
    return best_params

# Define a boxcar filter function
def boxcar_filter(data, size):
    kernel = np.ones((size, size)) / (size * size)
    from scipy.signal import convolve2d
    return convolve2d(data, kernel, mode="same", boundary="fill", fillvalue=np.nan)


# ### LOAD DATA WRs #####

wr_original_series = pd.read_csv(path_wrs,\
                index_col=0,names=['week0','dist'],skiprows=1,parse_dates=True)
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

for ivar in range(len(list_vars)):
    print(ivar,list_vars[ivar])

indices_vars_atm = [8,21,22,23]
indices_vars_ocn = [0,1,2,3,4,5,6,7,10,11,12]
indices_vars_lnd = [9,13,14,15,16,17,18,19,20]

dic_indices = {'atm':indices_vars_atm,
              'ocn':indices_vars_ocn,
              'lnd':indices_vars_lnd,
              'all':np.arange(len(list_vars))}

indices_vars = dic_indices[component]
########## JOIN DATA FROM COMPONENT ###############

all_dfs = []

for ivar in indices_vars:
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
    all_dfs.append(combined_df)

full_df = pd.concat(all_dfs,axis=1)
full_df['day_sin'] = np.sin(2 * np.pi * full_df.index.day_of_year / 365)
full_df['day_cos'] = np.cos(2 * np.pi * full_df.index.day_of_year / 365)

# ### RUN OPTIMIZATION ####

for week_out in range(1,9):
    start_time = datetime.now()
    print(f'WEEK: {week_out}')
    week_out_str = f'week{week_out}'

    fully_combined_df = pd.concat([full_df,df_shifts[week_out_str]],axis=1)
    fully_combined_df = fully_combined_df.dropna()

    dic_trainval, dic_test = get_train_val_test_periods(fully_combined_df)
    start_of_test_periods = np.arange(1981,2021,10)
    
    df_week_forecast = df_shifts[[week_out_str]].dropna()
    
    random_forecast = generate_random_forecast(df_week_forecast,
                                               seed_value=42)
    climatology_forecast = generate_random_forecast_with_monthly_probabilities(df_week_forecast, 
                                                                               seed_value=42)
    random_forecast_probs = generate_random_forecast_probabilities(df_week_forecast)
    climatology_forecast_probs = generate_probability_forecast_with_monthly_probabilities(df_week_forecast)

    list_results = []
    list_results_probs = []
    shap_results = []
    
    for iperiod in range(len(start_of_test_periods)):
        print(iperiod)
        X_trainval = dic_trainval[start_of_test_periods[iperiod]].iloc[:,:-1].values
        y_trainval = dic_trainval[start_of_test_periods[iperiod]].iloc[:,-1]
        
        X_test = dic_test[start_of_test_periods[iperiod]].iloc[:,:-1].values
        y_test = dic_test[start_of_test_periods[iperiod]].iloc[:,-1]
        best_params = optimize_xgboost(X_trainval,
               y_trainval,
               f'{path_hyperparams}/df_hyperparams_{list_vars[ivar]}_{week_out_str}_{iperiod}.csv')
        with open(f'{path_hyperparams}/besthyperparams_{list_vars[ivar]}_{week_out_str}_{iperiod}.json', 'w') as json_file:
            json.dump(best_params, json_file)
            
        #### HERE DEFINITION OF THE MODEL ####
        
        cw = class_weight.compute_sample_weight(
            class_weight='balanced',
            y=y_trainval
        )
        cw = cw**best_params['beta_class_weights']
        
        model = xgb.XGBClassifier(n_estimators=20,
                            max_depth=int(best_params['max_depth']),
                            learning_rate=10**best_params['log10_learning_rate'],
                            subsample=best_params['subsample'],
                            colsample_bytree=best_params['colsample_bytree'],
                            colsample_bylevel=best_params['colsample_bylevel'],
                            gamma=best_params['gamma'],
                            reg_alpha=10**best_params['log10_reg_alpha'],
                            reg_lambda=10**best_params['log10_reg_lambda'],
                            num_class=5,
                            objective = "multi:softprob",
                            tree_method='hist',
                            device = f'cuda:{gpu_id}')
        
        model.fit(X_trainval, y_trainval, sample_weight=cw)
        y_predicted = model.predict(X_test)
        print(f1_score(y_test,y_predicted,average='micro'))
        y_predicted_probs = model.predict_proba(X_test)
        y_predicted_probs = pd.DataFrame(y_predicted_probs,index=y_test.index)
        df_results_temp = pd.DataFrame(np.array([y_test.values,y_predicted]).T,
                                       index=y_test.index,
                                       columns=['y_true','y_predicted'])
        list_results.append(df_results_temp)
        list_results_probs.append(y_predicted_probs)
        
    df_results_full = pd.concat(list_results,axis=0)        
    df_results_probs_full = pd.concat(list_results_probs,axis=0)
    
    df_results_full.to_csv(f'{path_results}/Results_{component}_{week_out_str}.csv')
    df_results_probs_full.to_csv(f'{path_results_probs}/Results_{component}_{week_out_str}.csv')
    
    print(f1_score(df_results_full['y_true'],
                 df_results_full['y_predicted'],
                 average='micro'))