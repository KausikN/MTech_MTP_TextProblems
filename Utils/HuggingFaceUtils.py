"""
Utils
"""

# Imports
from .Utils import *

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from datasets import load_dataset, load_from_disk

# Main Functions
# HuggingFace Functions
## HuggingFace - Model Functions
def HuggingFace_Model_Load(
        model_path, 
        params={
            "model": {},
            "tokenizer": {},
            "config": {}
        }
    ):
    '''
    HuggingFace - Model - Load Pretrained
    '''
    # Load
    OUT = {
        "model": AutoModelForSequenceClassification.from_pretrained(
            model_path,
            **params["model"]
        ),
        "tokenizer": AutoTokenizer.from_pretrained(
            model_path,
            **params["tokenizer"]
        ),
        "config": AutoConfig.from_pretrained(
            model_path,
            **params["config"]
        )
    }

    return OUT

def HuggingFace_Model_Save(
        model_dir,
        DATA
    ):
    '''
    HuggingFace - Model - Save

    DATA must contain:
        - model
        - tokenizer
        - config
    '''
    # Init
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    PATHS = {
        "model": os.path.join(model_dir, "model"),
        "tokenizer": os.path.join(model_dir, "tokenizer"),
        "config": os.path.join(model_dir, "config")
    }
    # Save
    DATA["model"].save_pretrained(PATHS["model"])
    DATA["tokenizer"].save_pretrained(PATHS["tokenizer"])
    DATA["config"].save_pretrained(PATHS["config"])

    return PATHS

## HuggingFace - Dataset Functions
def HuggingFace_Dataset_LoadPretrained(
        dataset_path, 
        params={
            "dataset": {}
        }
    ):
    '''
    HuggingFace - Dataset - Load Pretrained
    '''
    # Load
    OUT = {
        "dataset": load_dataset(
            dataset_path,
            **params["dataset"]
        )
    }

    return OUT

def HuggingFace_Dataset_LoadLocal(
        dataset_path, 
        params={
            "dataset": {}
        }
    ):
    '''
    HuggingFace - Dataset - Load Local
    '''
    # Init
    PATHS = {
        "dataset": os.path.join(dataset_path, "dataset")
    }
    # Load
    OUT = {
        "dataset": load_from_disk(
            PATHS["dataset"],
            **params["dataset"]
        )
    }

    return OUT

def HuggingFace_Dataset_SaveLocal(
        dataset_dir,
        DATA
    ):
    '''
    HuggingFace - Dataset - Save Local

    DATA must contain:
        - dataset
    '''
    # Init
    if not os.path.exists(dataset_dir): os.makedirs(dataset_dir)
    PATHS = {
        "dataset": os.path.join(dataset_dir, "dataset")
    }
    # Save
    DATA["dataset"].save_to_disk(PATHS["dataset"])

    return PATHS