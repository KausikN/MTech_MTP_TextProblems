"""
Dataset Utils for IMDB Dataset

Link: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Expected Files in Dataset Folder:
    - IMDB Dataset.csv    :  Dataset CSV File
"""

# Imports
import os
import functools
import numpy as np
import pandas as pd
# Import from Parent Path
from Utils.KaggleUtils import *
from Utils.EncodeUtils import *

# Main Vars
DATASET_PATH = "Data/Datasets/IMDB/Data/"
DATASET_ITEMPATHS = {
    "kaggle": "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
    "test": "IMDB Dataset.csv"
}
DATASET_DATA = {
    "Sentiment Analysis": {
        "output_type": {
            "type": "category",
            "categories": ["negative", "positive"]
        },
        "cols": {
            "all": ["review", "sentiment"],
            "keep": ["review"],
            "keep_default": ["review"],
            "target": "sentiment"
        }
    }
}
DATASET_PARAMS = {
    "load": {
        "N_subset": 0.01
    },
    "encode": {	
        
    }
}
DATASET_SESSION_DATA = {}

# Main Functions
# Load Functions
def DatasetUtils_LoadCSV(path):
    '''
    DatasetUtils - Load CSV
    '''
    return pd.read_csv(path)

# Dataset Functions
def DatasetUtils_LoadDataset(
    path=DATASET_PATH, mode="test", 
    DATASET_ITEMPATHS=DATASET_ITEMPATHS, 
    task="Sentiment Analysis",
    other_params=DATASET_PARAMS,
    N=-1, 

    keep_cols=None, 
    **params
    ):
    '''
    DatasetUtils - Load Dataset
    '''
    # Init
    OTHER_PARAMS = other_params["load"]
    DatasetData = DATASET_DATA[task]
    if keep_cols is None: keep_cols = DatasetData["cols"]["keep"]
    csv_path = os.path.join(path, DATASET_ITEMPATHS[mode])
    # Download Dataset
    if not os.path.exists(csv_path):
        os.makedirs(DATASET_PATH, exist_ok=True)
        KaggleUtils_DownloadDataset(DATASET_ITEMPATHS["kaggle"], DATASET_PATH, quiet=False, unzip=True)
    # Get Dataset
    dataset = DatasetUtils_LoadCSV(csv_path)
    # Take N range
    if OTHER_PARAMS["N_subset"] < 1.0: dataset = dataset.iloc[::int(1.0/OTHER_PARAMS["N_subset"])]
    if type(N) == int:
        if N > 0: dataset = dataset.head(N)
    elif type(N) == list:
        if len(N) == 2: dataset = dataset.iloc[N[0]:N[1]]
    # Reset Columns
    dataset.columns = DatasetData["cols"]["all"]
    # Remove NaN values
    dataset.dropna(inplace=True)
    dataset.reset_index(drop=True, inplace=True)

    # Target Dataset
    dataset_target = None
    if DatasetData["cols"]["target"] is not None:
        dataset_target = dataset[[DatasetData["cols"]["target"]]].copy()
    
    # Input Dataset
    dataset_input = dataset[keep_cols].copy()

    # Return
    DATASET = {
        "task": task,
        "N": dataset.shape[0],
        "feature_names": {
            "input": list(dataset_input.columns) if dataset_input is not None else [],
            "target": list(dataset_target.columns) if dataset_target is not None else []
        },
        "target": dataset_target,
        "input": dataset_input,
        
        "session_params": {
            "init_session": ((type(N) == int) and (N == -1)),
            "other_params": other_params
        }
    }
    return DATASET

# Encode Functions
def DatasetUtils_EncodeDataset(
    dataset, 
    **params
    ):
    '''
    DatasetUtils - Encode Dataset
    '''
    global DATASET_SESSION_DATA
    # Init
    ## Data Init
    Fs = {}
    Ls = None
    FEATURES_INFO = {}
    ## Params Init
    OTHER_PARAMS = dataset["session_params"]["other_params"]["encode"]
    INIT_SESSION = dataset["session_params"]["init_session"]
    if INIT_SESSION:
        DATASET_SESSION_DATA = {
            
        }

    # Target Dataset
    Ls = np.zeros((dataset["N"], 2), dtype=float)
    if dataset["target"] is not None:
        ## Encode Target
        TargetMask = np.array(dataset["target"].to_numpy() == "positive", dtype=float)
        Ls = np.stack([1.0 - TargetMask, TargetMask], axis=1).reshape((-1, Ls.shape[1]))
        ## Finalize
        FEATURES_INFO["target"] = {
            "name": "sentiment",
            "type": {
                "type": "category",
                "categories": ["negative", "positive"]
            }
        }

    # Input Dataset
    dataset_input = dataset["input"]
    if dataset_input is not None:
        ## Init
        input_features_info = [
            {
                "name": dataset_input.columns[i],
                "type": {
                    "type": "text"
                }
            } for i in range(len(dataset_input.columns))
        ]
        ## Encode Dataset
        dataset_input = dataset_input.copy()
        ## Convert to numpy array
        dataset_input = dataset_input.to_numpy().astype(object)   
        ## Finalize
        Fs["input"] = dataset_input
        FEATURES_INFO["input"] = input_features_info

    # Return
    return Fs, Ls, FEATURES_INFO

# Display Functions
def DatasetUtils_DisplayDataset(
    dataset, 
    N=-1,
    **params
    ):
    '''
    DatasetUtils - Display Dataset
    '''
    # Init
    pass
    # Generate Display Data
    display_data = pd.DataFrame()
    for k in ["input", "target"]:
        if dataset[k] is not None:
            d = dataset[k]
            ## Take N range
            if type(N) == int:
                if N > 0: d = d.head(N)
            elif type(N) == list:
                if len(N) == 2: d = d.iloc[N[0]:N[1]]
            ## Concat
            display_data = pd.concat([display_data, d], axis=1)

    return display_data

# Main Vars
DATASET_FUNCS = {
    "full": functools.partial(DatasetUtils_LoadDataset, mode="test", DATASET_ITEMPATHS=DATASET_ITEMPATHS),
    "train": functools.partial(DatasetUtils_LoadDataset, mode="test", DATASET_ITEMPATHS=DATASET_ITEMPATHS),
    "val": functools.partial(DatasetUtils_LoadDataset, mode="test", DATASET_ITEMPATHS=DATASET_ITEMPATHS),
    "test": functools.partial(DatasetUtils_LoadDataset, mode="test", DATASET_ITEMPATHS=DATASET_ITEMPATHS),

    "encode": functools.partial(DatasetUtils_EncodeDataset),
    "display": functools.partial(DatasetUtils_DisplayDataset)
}