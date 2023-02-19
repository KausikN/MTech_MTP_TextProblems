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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

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
    def __init__(self, inputs, targets, tokenizer, max_len):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, item):
        text = str(self.inputs[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        pad_to_max_length=False,
        return_attention_mask=True,
        return_tensors="pt",
        )

        input_ids = pad_sequences(
            encoding["input_ids"], 
            maxlen=self.max_len, dtype=torch.Tensor, truncating="post", padding="post"
        )
        input_ids = input_ids.astype(dtype = "int64")
        input_ids = torch.tensor(input_ids) 

        attention_mask = pad_sequences(
            encoding["attention_mask"], 
            maxlen=self.max_len, dtype=torch.Tensor, truncating="post", padding="post"
        )
        attention_mask = attention_mask.astype(dtype="int64")
        attention_mask = torch.tensor(attention_mask)       

        return {
            "input": text,
            "input_ids": input_ids,
            "attention_mask": attention_mask.flatten(),
            "targets": torch.tensor(target, dtype=torch.long)
        }

class TextProblems_SentimentAnalysis_Base:
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

    base_params={
        "tokenizer": None,
        "pretrained_model": None,
        "dataset_loader": None
    },

    **params
    ):
        '''
        Text Problems - Sentiment Analysis

        Params:
         - n_classes : Number of classes
         - dataset_params : Dataset Parameters
         - train_params : Training Parameters
         - model_params : Model Parameters
         - random_state : Random State

        '''
        self.n_classes = n_classes
        self.dataset_params = dataset_params
        self.train_params = train_params
        self.model_params = model_params
        self.random_state = random_state
        self.base_params = base_params
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
        # Tokenizer
        self.tokenizer = self.base_params["tokenizer"]
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Features Info
        self.features_info = {
            "input": [],
            "target": {
                "name": "Sentiment",
                "type": {
                    "type": "category",
                    "categories": ["Negative", "Positive"]
                }
            }
        }
        # Model
        self.model = {"model": None}
        if self.model_params["load_path"] is not None:
            self.model["model"] = torch.load(self.model_params["load_path"])
        elif self.model_params["load_pretrained"]:
            self.model["model"] = self.base_params["pretrained_model"](num_labels=self.n_classes)

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
        # Init
        self.time_data["train"] = Time_Record("Model - Train")
        ## Features Info Init
        self.features_info = features_info
        ## Data Init
        Fs = np.array(Fs["input"])
        Ls = np.array(Ls)
        N_CLASSES = self.n_classes
        if Ls.shape[1] != N_CLASSES: raise Exception("Ls shape does not match n_classes")
        # Preprocess
        DATASETS = {"Fs": {}, "Ls": {}}
        DATASETS["Fs"]["train"], DATASETS["Fs"]["test"], DATASETS["Ls"]["train"], DATASETS["Ls"]["test"] = train_test_split(
            Fs, Ls, 
            test_size=self.dataset_params["test_size"], random_state=self.dataset_params["random_state"]
        )
        DATASETS["Fs"]["train"], DATASETS["Fs"]["val"], DATASETS["Ls"]["train"], DATASETS["Ls"]["val"] = train_test_split(
            DATASETS["Fs"]["train"], DATASETS["Ls"]["train"], 
            test_size=self.dataset_params["val_size"], random_state=self.dataset_params["random_state"]
        )
        DATASET_LOADERS = {}
        for dk in DATASETS["Fs"].keys():
            D = self.base_params["dataset_loader"](
                DATASETS["Fs"][dk], DATASETS["Ls"][dk], 
                self.tokenizer, self.dataset_params["max_len"]
            )
            DL = DataLoader(
                D, 
                batch_size=self.dataset_params["batch_size"],
                num_workers=4
            )
            DATASET_LOADERS[dk] = DL
        self.time_data["train"] = Time_Record("Data Preprocess", self.time_data["train"])
        # Train
        ## Init Model
        MODEL = self.model["model"]
        if self.model["model"] is None:
            self.model["model"] = self.base_params["pretrained_model"](num_labels=self.n_classes)
        MODEL = MODEL.to(self.device)
        ## Init Optimizer
        PARAM_OPTIM = list(MODEL.named_parameters())
        NO_DECAY = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in PARAM_OPTIM if not any(nd in n for nd in NO_DECAY)], "weight_decay": 0.01},
            {"params": [p for n, p in PARAM_OPTIM if any(nd in n for nd in NO_DECAY)], "weight_decay": 0.0}
        ]
        OPTIMIZER = AdamW(optimizer_grouped_parameters, lr=self.train_params["learning_rate"])
        TOTAL_STEPS = len(DATASET_LOADERS["train"]) * self.train_params["epochs"]
        SCHEDULER = get_linear_schedule_with_warmup(
            OPTIMIZER,
            num_warmup_steps=0,
            num_training_steps=TOTAL_STEPS
        )
        ## Train
        HISTORY = []
        OVERALL_HISTORY = {
            "train": {},
            "val": {}
        }
        for epoch in tqdm(range(self.train_params["epochs"])):
            ## Train
            TrainData = TrainModel_Epoch(
                MODEL,
                DATASET_LOADERS["train"],     
                OPTIMIZER, 
                SCHEDULER, 
                self.device
            )
            MODEL = TrainData["model"]
            ## Eval
            ValData = EvalModel_Basic(
                MODEL,
                DATASET_LOADERS["val"], 
                self.device
            )
            ## Record
            N_CUREPOCH_HISTORY_TRAIN = TrainData["counter"]
            N_CUREPOCH_HISTORY_VAL = ValData["counter"]
            CUR_HISTORY = {
                "epoch": epoch,
                "train_count": TrainData["counter"],
                "train": {},
                "val": {},
                "train_expanded": TrainData["metrics"]
            }
            for k in TrainData["metrics"][0].keys():
                CUR_HISTORY["train"][k] = np.mean([TrainData["metrics"][i][k] for i in range(N_CUREPOCH_HISTORY_TRAIN)])
            for k in ValData["metrics"][0].keys():
                CUR_HISTORY["val"][k] = np.mean([ValData["metrics"][i][k] for i in range(N_CUREPOCH_HISTORY_VAL)])
            HISTORY.append(CUR_HISTORY)
            ## Print
            print("Epoch: ", epoch)
            print("Train Loss: ", CUR_HISTORY["train"]["loss"])
            print("Train Acc: ", CUR_HISTORY["train"]["acc"])
            print("Val Loss: ", CUR_HISTORY["val"]["loss"])
            print("Val Acc: ", CUR_HISTORY["val"]["acc"])
            ## Save Best Model
            if len(OVERALL_HISTORY["val"].keys()) == 0 or OVERALL_HISTORY["val"]["accuracy"] < CUR_HISTORY["val"]["accuracy"]:
                OVERALL_HISTORY = CUR_HISTORY
                torch.save(MODEL.state_dict(), os.path.join(self.train_params["save_dir"], "best_model.bin"))
        self.time_data["train"] = Time_Record("Model Training", self.time_data["train"])
        # Record
        self.model = {
            "model": MODEL
        }
        self.history = {
            "history": HISTORY,
            "best_model_info": OVERALL_HISTORY
        }
        self.time_data["train"] = Time_Record("", self.time_data["train"], finish=True)

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
        Plots = {
            "Train Loss": None,
            "Train Accuracy": None,
            "Val Loss": None,
            "Val Accuracy": None
        }
        Data = {}
        # Get Data
        
        # Plot
        if not disable_plots:
            ## Train Loss
            Plots["Train Loss"] = plt.figure()
            plt.plot([self.history["history"][i]["train"]["loss"] for i in range(len(self.history["train"]))])
            plt.title("Train Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            ## Train Accuracy
            Plots["Train Accuracy"] = plt.figure()
            plt.plot([self.history["history"][i]["train"]["accuracy"] for i in range(len(self.history["train"]))])
            plt.title("Train Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            ## Val Loss
            Plots["Val Loss"] = plt.figure()
            plt.plot([self.history["history"][i]["val"]["loss"] for i in range(len(self.history["train"]))])
            plt.title("Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            ## Val Accuracy
            Plots["Val Accuracy"] = plt.figure()
            plt.plot([self.history["history"][i]["val"]["accuracy"] for i in range(len(self.history["train"]))])
            plt.title("Val Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
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
        Fs = np.array(Fs["input"])
        Ls = np.array(Ls)
        # Predict
        MODEL = self.model["model"]
        DATA_LOADER_TEST = self.base_params["dataset_loader"](
            Fs, Ls,
            self.tokenizer, self.dataset_params["max_len"]
        )
        TestData = EvalModel_Basic(
            MODEL,
            DATA_LOADER_TEST,
            self.device
        )
        # Record
        N_CUREPOCH_HISTORY_TEST = TestData["counter"]
        METRICS = {
            k: np.mean([TestData["metrics"][i][k] for i in range(N_CUREPOCH_HISTORY_TEST)])
            for k in TestData["metrics"][0].keys()
        }

        return METRICS

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
        # Init
        N_CLASSES = self.n_classes
        Fs = np.array(Fs["input"])
        Ls = np.zeros((Fs.shape[0], N_CLASSES))
        MAX_LEN = self.dataset_params["max_len"]
        MODEL = self.model["model"]
        # Preprocess
        ## Init
        INPUT_IDS =[]
        ATTENTION_MASKS = []
        ## Encode
        for i in range(Fs.shape[0]):
            F_encoded = self.tokenizer.encode_plus(
                Fs[i][0],
                max_length=MAX_LEN,
                add_special_tokens=True,
                return_token_type_ids=False,
                pad_to_max_length=False,
                return_attention_mask=True,
                return_tensors="pt",
            )
            ### Input IDs
            input_ids = pad_sequences(
                F_encoded["input_ids"], 
                maxlen=MAX_LEN, dtype=torch.Tensor, truncating="post", padding="post"
            ).astype(dtype="int64")
            input_ids = torch.tensor(input_ids)
            INPUT_IDS.append(input_ids)
            ### Attention Mask
            attention_mask = pad_sequences(
                F_encoded["attention_mask"], 
                maxlen=MAX_LEN, dtype=torch.Tensor, truncating="post", padding="post"
            ).astype(dtype="int64")
            attention_mask = torch.tensor(attention_mask)
            ATTENTION_MASKS.append(attention_mask)
        ## Finalize
        INPUT_IDS = torch.cat(INPUT_IDS, dim=0)
        ATTENTION_MASKS = torch.cat(ATTENTION_MASKS, dim=0)
        INPUT_IDS = INPUT_IDS.reshape(-1, MAX_LEN).to(self.device)
        ATTENTION_MASKS = ATTENTION_MASKS.reshape(-1, MAX_LEN).to(self.device)
        # Predict
        outputs = MODEL(input_ids=INPUT_IDS, attention_mask=ATTENTION_MASKS)
        PROB_DIST = F.softmax(outputs.logits, dim=-1).cpu().detach().numpy().tolist()
        Ls = PROB_DIST

        return Ls
    
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