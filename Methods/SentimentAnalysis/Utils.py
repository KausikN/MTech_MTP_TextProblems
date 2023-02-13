"""
Utils
"""

# Imports
import os
import time
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# TQDM
CONFIG = json.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "config.json"), "r"))
if CONFIG["tqdm_notebook"]:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# Main Classes
class TextProblems_SentimentAnalysis_Base:
    def __init__(self,
    model=None,
    **params
    ):
        '''
        Text Problems - Sentiment Analysis - Algorithm

        Params:
         - model : Pretained model
        '''
        self.model = model
        self.__dict__.update(params)

    def train(self,
        Fs, Ls, 
        features_info={"input": [], "target": {}},
        **params
        ):
        '''
        Train

        Train algorithm on given features and labels.

        Inputs:
            - Fs : Text Input (N_Samples, 1)
            - Ls : Label Distribution (N_Samples, Label_Dim)
            
        Outputs:
            - model : Model that can be used to predict labels from features
        '''
        return None
        
    def visualise(self):
        '''
        Visualise
        '''
        return {}

    def predict(self,
        Fs, 

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
        return None
    
    def save(self, path):
        '''
        Save
        '''
        # Init
        path_data = os.path.join(path, "data.p")
        data = self.__dict__
        # Save
        with open(path_data, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        '''
        Load
        '''
        # Init
        path_data = os.path.join(path, "data.p")
        # Load
        with open(path_data, "rb") as f:
            data = pickle.load(f)
        # Update
        self.__dict__.update(data)

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

# Plot Functions


# Evaluation Functions
# def Eval_Basic(labels_true, labels_pred, unique_labels=None):
#     '''
#     Eval - Basic
#     '''
#     evals = {
#         "confusion_matrix": confusion_matrix(labels_true, labels_pred, labels=unique_labels),
#         "classification_report": classification_report(labels_true, labels_pred, labels=unique_labels),
#         "accuracy_score": accuracy_score(labels_true, labels_pred)
#     }

#     return evals

# Train / Eval Functions
def TrainModel_Epoch(model, data_loader, optimizer, device, scheduler):
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

def EvalModel_Basic(model, data_loader, device):
    '''
    Eval Model - Basic
    '''
    # Init
    model = model.eval()
    METRICS = []
    COUNTER = 0
    # Loop
    with torch.no_grad():
        for d in data_loader:
            ## Init
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            ## Forward
            outputs = model(
                input_ids=input_ids, token_type_ids=None, 
                attention_mask=attention_mask, labels=targets
            )
            loss = outputs[0]
            logits = outputs[1]
            _, prediction = torch.max(outputs[1], dim=1)
            targets = targets.cpu().detach().numpy()
            prediction = prediction.cpu().detach().numpy()
            ## Metrics
            METRICS.append({
                "loss": loss.item(),
                "accuracy": accuracy_score(targets, prediction)
            })
            ## Update
            COUNTER += 1

    return {
        "counter": COUNTER,
        "metrics": METRICS
    }