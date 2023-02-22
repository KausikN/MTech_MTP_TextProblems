"""
Utils
"""

# Imports
import os
import time
import json
import pickle
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler

from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AdamW, get_linear_schedule_with_warmup

# TQDM
CONFIG = json.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "config.json"), "r"))
if CONFIG["tqdm_notebook"]:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# Main Classes
class DatasetLoader_SentimentAnalysis_Base(Dataset):
    def __init__(self, inputs, targets, tokenizer, **params):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.__dict__.update(params)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, item):
        ## Init
        text = str(self.inputs[item])
        target = self.targets[item]
        ## Encode
        encoding = self.tokenizer.encode(text)

        return {
            "input": text,
            "input_ids": np.array(encoding["input_ids"], dtype=np.long),
            "targets": np.array(target, dtype=np.long)
        }

class TextProblems_SentimentAnalysis_Base:
    def __init__(self,
    n_classes=2,
    dataset_params={},
    train_params={},
    predict_params={},
    model_params={},
    random_state=0,
    **params
    ):
        '''
        Text Problems - Sentiment Analysis

        Params:
         - n_classes : Number of classes
         - dataset_params : Dataset Parameters
         - train_params : Training Parameters
         - predict_params : Prediction Parameters
         - model_params : Model Parameters
         - random_state : Random State

        '''
        self.n_classes = n_classes
        self.dataset_params = dataset_params
        self.train_params = train_params
        self.predict_params = predict_params
        self.model_params = model_params
        self.random_state = random_state
        self.__dict__.update(params)
        # Time Params
        self.time_data = {
            "train": {},
            "predict": {}
        }
        # History
        self.history = {
            "history": [],
            "best_model_info": None
        }
        # Features Info
        self.features_info = {
            "input": [],
            "target": {
                "name": "Sentiment",
                "type": {
                    "type": "category",
                    "categories": ["negative", "positive"]
                }
            }
        }
        # Save/Load Params
        self.save_paths = {
            "class_obj": "class_obj.p"
        }

    def train(self,
        Fs, Ls, 
        features_info={"input": [], "target": {}},
        **params
        ):
        '''
        Train

        Train Model on given features and labels.

        Inputs:
            - Fs : Text Input (N_Samples, 1)
            - Ls : Label Distribution (N_Samples, Label_Dim)
            
        Outputs:
            - model : Model that can be used to predict labels from features
        '''
        return self.model
        
    def visualise(self, disable_plots=False):
        '''
        Visualise
        '''
        # Init
        VisData = {
            "figs": {
                "pyplot": {},
                "plotly_chart": {}
            },
            "data": {}
        }
        Plots = {}
        Data = {}
        ## Record
        Data["Time"] = self.time_data
        ## CleanUp
        for k in Plots.keys():
            if Plots[k] is not None:
                plt.close(Plots[k])
        # Record
        VisData["figs"]["plotly_chart"] = Plots
        VisData["data"] = Data

        return VisData

    def tokenize(self,
        texts,
        **params
        ):
        '''
        Tokenize

        Tokenize the text features.

        Inputs:
            - texts : Text Input (N_Samples, 1)
        Outputs:
            - input_ids : Tokenized Text Input (N_Samples, MAX_LEN)
            - OtherData : Other Data (Attention Masks, etc)
        '''
        # Init
        TOKEN_DATA = {
            "input_ids": []
        }

        return TOKEN_DATA

    def test(self,
        Fs, Ls,
        **params
        ):
        '''
        Predict

        Test Model on given features and labels.

        Inputs:
            - Fs : Text Input (N_Samples, 1)
            - Ls : True Label Distributions (N_Samples, Label_Dim)
        Outputs:
            - Metrics : Test Metrics
        '''
        # Init
        Ls = np.array(Ls)
        # Predict
        Ls_pred = self.predict(Fs, record_time=False)
        # Metrics
        Ls_indices = np.argmax(Ls, axis=-1)
        Ls_pred_indices = np.argmax(Ls_pred, axis=-1)
        METRICS = Eval_Basic(Ls_indices, Ls_pred_indices)

        return METRICS

    def predict(self,
        Fs, 
        record_time=True,
        **params
        ):
        '''
        Predict

        Predict labels from features.

        Inputs:
            - Fs : Text Input (N_Samples, 1)
        Outputs:
            - Ls : Label Distributions (N_Samples, Label_Dim)
        '''
        # Init
        Ls = None

        return Ls
    
    def save(self, path):
        '''
        Save
        '''
        # Init
        paths = {
            "class_obj": os.path.join(path, self.save_paths["class_obj"]),
        }
        # Save Class Object (Ignore some objects while saving)
        dict_data = self.__dict__
        ## Save
        with open(paths["class_obj"], "wb") as f: pickle.dump(dict_data, f)

    def load(self, path):
        '''
        Load
        '''
        # Init
        paths = {
            "class_obj": os.path.join(path, self.save_paths["class_obj"])
        }
        # Load Class Object
        with open(paths["class_obj"], "rb") as f: dict_data = pickle.load(f)
        # Load
        self.__dict__.update(dict_data)

# Main Functions
# Array Functions

# Time Functions
def Time_Record(name, time_data=None, finish=False):
    '''
    Time - Record
    '''
    # Init
    if time_data is None:
        curtime = time.time()
        time_data = {
            "overall": {
                "title": name
            },
            "current": {
                "prev": curtime,
                "cur": curtime
            },
            "record": []
        }
        return time_data
    # Finish
    if finish:
        time_data["current"]["cur"] = time.time()
        time_data["overall"].update({
            "time": sum([i["time"] for i in time_data["record"]]),
        })
        del time_data["current"]
        return time_data
    # Record
    time_data["current"]["cur"] = time.time()
    time_data["record"].append({
        "name": name,
        "time": time_data["current"]["cur"] - time_data["current"]["prev"]
    })
    time_data["current"]["prev"] = time_data["current"]["cur"]
    return time_data

def Time_Combine(name, time_datas):
    '''
    Time - Combine
    '''
    # Init
    combined_time_data = {
        "overall": {
            "title": name
        },
        "record": []
    }
    # Combine
    for time_data_k in time_datas.keys():
        combined_time_data["record"].append({
            "name": time_data_k,
            "time": sum([i["overall"]["time"] for i in time_datas[time_data_k]]),
        })
    combined_time_data["overall"].update({
        "time": sum([i["time"] for i in combined_time_data["record"]]),
    })

    return combined_time_data

# Plot Functions


# Evaluation Functions
def Eval_Basic(labels_true, labels_pred, unique_labels=None):
    '''
    Eval - Basic
    '''
    evals = {
        "counter": labels_true.shape[0],
        "metrics": {
            "confusion_matrix": confusion_matrix(labels_true, labels_pred, labels=unique_labels).tolist(),
            "classification_report": classification_report(labels_true, labels_pred, labels=unique_labels),
            "accuracy_score": accuracy_score(labels_true, labels_pred),
            "precision_score": precision_score(labels_true, labels_pred, average="weighted"),
            "recall_score": recall_score(labels_true, labels_pred, average="weighted"),
            "f1_score": f1_score(labels_true, labels_pred, average="weighted")
        }
    }

    return evals

# Train / Eval Functions
def TrainModel_Epoch(model, data_loader, optimizer, scheduler, device):
    '''
    Train Model - Epoch
    '''
    # Init
    model = model.train()
    METRICS = []
    COUNTER = 0
    # Loop
    for d in data_loader:
        ## Init
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        ## Forward
        outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels = targets)
        loss = outputs[0]
        logits = outputs[1]
        # preds = preds.cpu().detach().numpy()
        _, prediction = torch.max(outputs[1], dim=1)
        targets = targets.cpu().detach().numpy()
        prediction = prediction.cpu().detach().numpy()
        ## Metrics
        METRICS.append({
            "loss": loss.item(),
            "accuracy": accuracy_score(targets, prediction)
        })
        ## Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        ## Update
        COUNTER += 1

    return {
        "counter": COUNTER,
        "metrics": METRICS,
        "model": model
    }