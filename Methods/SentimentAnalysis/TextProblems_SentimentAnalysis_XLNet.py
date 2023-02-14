"""
Text Problems - Sentiment Analysis - XLNet

References:
https://medium.com/swlh/using-xlnet-for-sentiment-classification-cfa948e65e85
"""

# Imports
from transformers import XLNetTokenizer, XLNetForSequenceClassification

from .Utils import *

# Main Classes
# XLNet
class TextProblems_SentimentAnalysis_XLNetBase(TextProblems_SentimentAnalysis_Base):
    def __init__(self,
    n_classes=2,
    dataset_params={
        "test_size": 0.2,
        "val_size": 0.05,
        "random_state": 0,
        "max_len": 16,
        "batch_size": 4
    },
    train_params={
        "batch_size": 4,
        "epochs": 3,
        "learning_rate": 3e-5,
        "save_dir": "_models/temp/"
    },
    model_params={
        "load_path": None,
        "load_pretrained": True
    },
    random_state=0,

    **params
    ):
        '''
        Text Problems - Sentiment Analysis - XLNet Base

        Params:
         - n_classes : Number of classes
         - dataset_params : Dataset Parameters
         - train_params : Training Parameters
         - model_params : Model Parameters
         - random_state : Random State

        '''
        # Init
        self.base_params = {
            "tokenizer": XLNetTokenizer.from_pretrained("xlnet-base-cased"),
            "pretrained_model": functools.partial(XLNetForSequenceClassification.from_pretrained, "xlnet-base-cased"),
            "dataset_loader": DatasetLoader_SentimentAnalysis_Base
        }
        # Call Parent
        super().__init__(
            n_classes=n_classes,
            dataset_params=dataset_params,
            train_params=train_params,
            model_params=model_params,
            random_state=random_state,
            base_params=self.base_params,
            **params
        )

# Main Vars
TASK_FUNCS = {
    "XLNet Base": {
        "class": TextProblems_SentimentAnalysis_XLNetBase,
        "params": {
            "n_classes": 2,
            "dataset_params": {
                "test_size": 0.2,
                "val_size": 0.05,
                "random_state": 0,
                "max_len": 16,
                "batch_size": 4
            },
            "train_params": {
                "batch_size": 4,
                "epochs": 3,
                "learning_rate": 3e-5,
                "save_dir": "_models/temp/"
            },
            "model_params": {
                "load_path": None,
                "load_pretrained": True
            },
            "random_state": 0
        }
    }
}