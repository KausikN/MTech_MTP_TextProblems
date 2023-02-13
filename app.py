"""
Streamlit App
"""

# Imports
import os
import time
import json
import pickle
import functools
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from TextProblems import *

# Main Vars
PATHS = {
    "temp": "Data/Temp/",
    "models": "_models/",
    "credentials_kaggle": "_credentials/kaggle.json",
    "settings": "_appdata/settings.json"
}
SETTINGS = {}

# Progress Classes
class ProgressBar:
    def __init__(self, title, max_value):
        self.max_value = max_value
        self.title = st.sidebar.empty()
        self.bar = st.sidebar.progress(0)
        self.value = -1
        self.update(title)

    def update(self, title):
        self.title.markdown(title)
        self.value += 1
        self.bar.progress(self.value / self.max_value)

    def finish(self):
        self.title.empty()
        self.bar.empty()

# Utils Functions
def name_to_path(name):
    # Convert to Lowercase
    name = name.lower()
    # Remove Special Chars
    for c in [" ", "-", ".", "<", ">", "/", "\\", ",", ";", ":", "'", '"', "|", "*", "?"]:
        name = name.replace(c, "_")

    return name

# Cache Data Functions
def CacheData_TrainedModel(
    USERINPUT_Method, dataset,
    keep_cols,
    **params
    ):
    '''
    Cache Data - Trained Model
    '''
    # Load Dataset
    DATASET_MODULE = DATASETS[dataset["name"]]
    DATASET = DATASET_MODULE.DATASET_FUNCS["test"](
        keep_cols=keep_cols,
        task=dataset["task"],
        other_params=dataset["params"]
    )
    Fs, Ls, FEATURES_INFO = DATASET_MODULE.DATASET_FUNCS["encode"](
        DATASET
    )
    # Init Model
    MODEL = USERINPUT_Method["class"](**USERINPUT_Method["params"])
    MODEL_PARAMS = {
        "features_info": FEATURES_INFO
    }
    # Train Model
    MODEL.train(Fs, Ls, **MODEL_PARAMS)

    return MODEL

# UI Functions
def UI_DisplayLabelDistribution(L, label_names=[]):
    '''
    Display Label Distribution
    '''
    # Init
    if len(label_names) == 0: label_names = ["C_"+str(i) for i in range(len(L))]
    # Display
    st.markdown("### Label Distribution")
    ## Plots
    fig = plt.figure()
    plt.bar(label_names, L)
    plt.title("Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Probability")
    plt.close(fig)
    st.pyplot(fig)
    ## Data
    data = {
        "Best Label": {
            "Label": label_names[np.argmax(L)],
            "Value": np.max(L)
        },
        "Worst Label": {
            "Label": label_names[np.argmin(L)],
            "Value": np.min(L)
        }
    }
    st.write(data)

def UI_DisplayVisData(OutData):
    '''
    Display Algorithm Visualisation Data
    '''
    # Init
    st.markdown("# Visualisations")
    st.markdown("## Plots")

    # Graphs
    for k in OutData["figs"]["plotly_chart"].keys():
        st.markdown(f"### {k}")
        cols = st.columns(len(OutData["figs"]["plotly_chart"][k]))
        for i in range(len(OutData["figs"]["plotly_chart"][k])):
            if SETTINGS["plots_interactive"]:
                cols[i].plotly_chart(OutData["figs"]["plotly_chart"][k][i])
            else:
                cols[i].pyplot(OutData["figs"]["plotly_chart"][k][i])
    # Plots
    for k in OutData["figs"]["pyplot"].keys():
        st.markdown(f"### {k}")
        cols = st.columns(len(OutData["figs"]["pyplot"][k]))
        for i in range(len(OutData["figs"]["pyplot"][k])):
            cols[i].pyplot(OutData["figs"]["pyplot"][k][i])
    # Data
    st.markdown("## Data")
    for k in OutData["data"].keys():
        st.markdown(f"### {k}")
        st.write(OutData["data"][k])

def UI_LoadDataset(TASK, dataset_params=None):
    '''
    Load Dataset
    '''
    st.markdown("## Load Dataset")
    # Select Dataset
    cols = st.columns((1, 3))
    DATASETS_AVAILABLE = [d for d in list(DATASETS.keys()) if TASK in DATASETS[d].DATASET_DATA.keys()]
    USERINPUT_Dataset = cols[0].selectbox("Select Dataset", DATASETS_AVAILABLE)
    DATASET_MODULE = DATASETS[USERINPUT_Dataset]
    ## Load Params
    if dataset_params is None:
        USERINPUT_DatasetParams_str = cols[1].text_area(
            "Params", 
            value=json.dumps(DATASET_MODULE.DATASET_PARAMS, indent=8),
            height=200
        )
        dataset_params = json.loads(USERINPUT_DatasetParams_str)
    # Load Dataset
    DATASET = DATASET_MODULE.DATASET_FUNCS["full"](
        task=TASK,
        other_params=dataset_params
    )
    N = DATASET["N"]

    # Options
    cols = st.columns(2)
    USERINPUT_Options = {
        "n_samples": cols[0].markdown(f"Count: **{N}**"),
        "display": cols[1].checkbox("Display Dataset", value=True)
    }

    # Display
    if USERINPUT_Options["display"]:
        USERINPUT_ViewSampleIndex = st.slider(f"View Sample ({N} Samples)", 0, N-1, 0, 1)
        DisplayData = DATASET_MODULE.DATASET_FUNCS["display"](DATASET, [USERINPUT_ViewSampleIndex, USERINPUT_ViewSampleIndex+1]).to_dict()
        st.table([{k: DisplayData[k][list(DisplayData[k].keys())[0]] for k in DisplayData.keys()}])

    DATA = {
        "name": USERINPUT_Dataset,
        "task": TASK,
        "module": DATASET_MODULE,
        "dataset": DATASET,
        "params": dataset_params
    }
    return DATA

def UI_TrainModel(DATA):
    '''
    Train Model
    '''
    st.markdown("## Train Model")
    # Load Method
    TASK = DATA["task"]
    TaskModules = TASK_MODULES[TASK]
    USERINPUT_Module = st.selectbox("Select Task Module", list(TaskModules.keys()))
    cols = st.columns((1, 3))
    USERINPUT_MethodName = cols[0].selectbox(
        "Select Task Method",
        list(TaskModules[USERINPUT_Module].keys())
    )
    USERINPUT_Method = TaskModules[USERINPUT_Module][USERINPUT_MethodName]
    # Load Params
    USERINPUT_Params_str = cols[1].text_area(
        "Params", 
        value=json.dumps(USERINPUT_Method["params"], indent=8),
        height=200
    )
    USERINPUT_Params = json.loads(USERINPUT_Params_str)
    USERINPUT_Method = {
        "class": USERINPUT_Method["class"],
        "params": USERINPUT_Params
    }
    USERINPUT_Params = json.loads(USERINPUT_Params_str) # Redo to create new object
    # Other Params
    DatasetData = DATA["module"].DATASET_DATA[TASK]
    KeepCols = DatasetData["cols"]["keep"]
    KeepCols_Default = DatasetData["cols"]["keep_default"]
    USERINPUT_Dataset = {
        "dataset": {
            "name": DATA["name"],
            "task": DATA["task"],
            "params": DATA["params"]
        }
    }
    USERINPUT_OtherParams = {
        "keep_cols": {
            k:  st.multiselect(
                    "Keep Columns: " + str(k), 
                    list(KeepCols[k]),
                    default=list(KeepCols_Default[k])
                )
            for k in KeepCols.keys()
        }
    }
    # Process Check
    USERINPUT_Process = st.checkbox("Stream Process", value=False)
    if not USERINPUT_Process: USERINPUT_Process = st.button("Process")
    if not USERINPUT_Process: st.stop()
    # Get Trained Model
    USERINPUT_Model = CacheData_TrainedModel(USERINPUT_Method, **USERINPUT_Dataset, **USERINPUT_OtherParams)
    # Update Data
    DATA["model_params"] = {
        "module_name": USERINPUT_Module,
        "method_name": USERINPUT_MethodName,
        "method_params": USERINPUT_Params
    }
    DATA["other_params"] = USERINPUT_OtherParams

    # Display Model Visualisations
    VisData = USERINPUT_Model.visualise()
    UI_DisplayVisData(VisData)

    return USERINPUT_Model, DATA

def UI_LoadTaskInput(DATA):
    '''
    Load Task Input
    '''
    # Init
    DATASET, DATASET_MODULE = DATA["dataset"], DATA["module"]
    # Load Input
    st.markdown("## Load Input")
    # Select Input
    N = DATASET["N"]
    USERINPUT_ViewSampleIndex = st.slider(f"Select Input ({N} Samples)", 0, N-1, 0, 1)
    DisplayData = DATASET_MODULE.DATASET_FUNCS["display"](DATASET, [USERINPUT_ViewSampleIndex, USERINPUT_ViewSampleIndex+1]).to_dict()
    st.table([{k: DisplayData[k][list(DisplayData[k].keys())[0]] for k in DisplayData.keys()}])
    # Encode Input
    InputData = DATASET_MODULE.DATASET_FUNCS["test"](
        N=[USERINPUT_ViewSampleIndex, USERINPUT_ViewSampleIndex+1],
        keep_cols=DATA["other_params"]["keep_cols"],
        task=DATA["task"],
        other_params=DATA["params"]
    )
    Input_Fs, Input_Ls, FEATURES_INFO = DATASET_MODULE.DATASET_FUNCS["encode"](InputData)
    USERINPUT_Input = {
        "F": Input_Fs,
        "L": Input_Ls,
        "features_info": FEATURES_INFO
    }

    return USERINPUT_Input

# Load / Save Model Functions
def Model_SaveModelData(USERINPUT_Model, DATA, suffix="1"):
    '''
    Model - Save Model and Dataset metadata
    '''
    # Init
    task_name = name_to_path(DATA["task"])
    data_name = name_to_path(DATA["name"])
    module_name = name_to_path(DATA["model_params"]["module_name"])
    method_name = name_to_path(DATA["model_params"]["method_name"])
    dir_path = os.path.join(PATHS["models"], task_name, data_name, module_name, method_name, suffix)
    # Create Dirs
    if not os.path.exists(dir_path): os.makedirs(dir_path)
    # Save Model Data
    USERINPUT_Model.save(dir_path)
    # Save Dataset Data
    save_params = {
        "data_name": DATA["name"],
        "task": DATA["task"],
        "dataset_params": DATA["params"],
        "model_params": DATA["model_params"],
        "other_params": DATA["other_params"]
    }
    json.dump(save_params, open(os.path.join(dir_path, "params.json"), "w"), indent=4)
    # Save Session Data
    pickle.dump(DATA["module"].DATASET_SESSION_DATA, open(os.path.join(dir_path, "session_data.p"), "wb"))

def Model_LoadModelData(path):
    '''
    Model - Load Model and Dataset metadata
    '''
    # Init
    # Check Exists
    if not os.path.exists(path): return None, None
    # Load Session Data
    session_data = pickle.load(open(os.path.join(path, "session_data.p"), "rb"))
    # Load Dataset Data
    load_params = json.load(open(os.path.join(path, "params.json"), "r"))
    load_params["model_params"]["method_params"]["model_params"]["load_path"] = ""
    # Load Model Data
    # Load Model Base
    USERINPUT_ModelBase = TASK_MODULES[load_params["task"]][load_params["model_params"]["module_name"]][load_params["model_params"]["method_name"]]
    USERINPUT_Model = USERINPUT_ModelBase["class"](**load_params["model_params"]["method_params"])
    USERINPUT_Model.load(path)

    return USERINPUT_Model, load_params, session_data

# Main Functions
def textproblems_init_basic(TASK="Sentiment Analysis"):
    # Title
    st.markdown(f"# Text Problems - {TASK} - Pretrained")

    # Load Inputs
    # Init
    PROGRESS_BARS = {
        "overall": ProgressBar("Started", 4)
    }
    USERINPUT_MODELSUFFIX = "pretrained"
    # Load Dataset
    PROGRESS_BARS["overall"].update("Loading Dataset...") # 1
    DATA = UI_LoadDataset(TASK)
    # Load Model
    PROGRESS_BARS["overall"].update("Loading Model...") # 2
    USERINPUT_Module = st.selectbox("Select Task Module", list(TASK_MODULES[TASK].keys()))
    cols = st.columns(2)
    USERINPUT_MethodName = cols[0].selectbox(
        "Select Task Method",
        list(TASK_MODULES[TASK][USERINPUT_Module].keys())
    )
    USERINPUT_Method = TASK_MODULES[TASK][USERINPUT_Module][USERINPUT_MethodName]
    # Load Params
    USERINPUT_Params_str = cols[1].text_area(
        "Params", 
        value=json.dumps(USERINPUT_Method["params"], indent=8),
        height=200
    )
    USERINPUT_Params = json.loads(USERINPUT_Params_str)
    USERINPUT_Method = {
        "class": USERINPUT_Method["class"],
        "params": USERINPUT_Params
    }
    # Process Check
    USERINPUT_Process = st.checkbox("Stream Process", value=False)
    if not USERINPUT_Process: USERINPUT_Process = st.button("Process")
    if not USERINPUT_Process: st.stop()
    # Init Model Params
    DATA["model_params"] = {
        "module_name": USERINPUT_Module,
        "method_name": USERINPUT_MethodName,
        "method_params": USERINPUT_Params
    }
    DATA["other_params"] = {
        "keep_cols": None
    }
    # Load Pretrained Model
    USERINPUT_Model = USERINPUT_Method["class"](**USERINPUT_Method["params"])
    # Save Model
    PROGRESS_BARS["overall"].update("Saving Model...") # 3
    Model_SaveModelData(USERINPUT_Model, DATA, suffix=USERINPUT_MODELSUFFIX)
    PROGRESS_BARS["overall"].update("Finished") # 4
    PROGRESS_BARS["overall"].finish()

def textproblems_train_basic(TASK="Sentiment Analysis"):
    # Title
    st.markdown(f"# Text Problems - {TASK} - Train")

    # Load Inputs
    # Init
    PROGRESS_BARS = {
        "overall": ProgressBar("Started", 4)
    }
    USERINPUT_MODELSUFFIX = st.sidebar.text_input("Model Save Suffix", value="")
    # Dataset
    PROGRESS_BARS["overall"].update("Loading Dataset...") # 1
    DATA = UI_LoadDataset(TASK)

    # Process Inputs
    # Train Model
    PROGRESS_BARS["overall"].update("Training Model...") # 2
    USERINPUT_Model, DATA = UI_TrainModel(DATA)
    # Save Model
    PROGRESS_BARS["overall"].update("Saving Model...") # 3
    Model_SaveModelData(USERINPUT_Model, DATA, suffix=USERINPUT_MODELSUFFIX)
    PROGRESS_BARS["overall"].update("Finished") # 4
    PROGRESS_BARS["overall"].finish()

def textproblems_test_basic(TASK="Sentiment Analysis"):
    # Title
    st.markdown(f"# Text Problems - {TASK} - Test")

    # Load Inputs
    # Init
    PROGRESS_BARS = {
        "overall": ProgressBar("Started", 7)
    }
    # Load Dataset
    PROGRESS_BARS["overall"].update("Loading Dataset...") # 1
    DATA = UI_LoadDataset(TASK)
    # Load Model
    PROGRESS_BARS["overall"].update("Loading Model...") # 2
    task_name = name_to_path(DATA["task"])
    data_name = name_to_path(DATA["name"])
    cols = st.columns(2)
    USERINPUT_Module = cols[0].selectbox("Select Task Module", list(TASK_MODULES[DATA["task"]].keys()))
    USERINPUT_MethodName = cols[1].selectbox(
        "Select Task Method",
        list(TASK_MODULES[DATA["task"]][USERINPUT_Module].keys())
    )
    module_name = name_to_path(USERINPUT_Module)
    method_name = name_to_path(USERINPUT_MethodName)
    parent_dir_path = os.path.join(PATHS["models"], task_name, data_name, module_name, method_name)
    suffix = st.selectbox("Select Model", os.listdir(parent_dir_path))
    dir_path = os.path.join(parent_dir_path, suffix)
    USERINPUT_Model, LOAD_PARAMS, SESSION_DATA = Model_LoadModelData(dir_path)
    if USERINPUT_Model is None:
        st.error("No Model Found")
        return
    DATA["module"].DATASET_SESSION_DATA = SESSION_DATA
    DATA["params"] = LOAD_PARAMS["dataset_params"]
    DATA["model_params"] = LOAD_PARAMS["model_params"]
    DATA["other_params"] = LOAD_PARAMS["other_params"]
    # Input
    PROGRESS_BARS["overall"].update("Loading Input...") # 3
    USERINPUT_TaskInput = UI_LoadTaskInput(DATA)

    # Process Inputs
    # Predict
    PROGRESS_BARS["overall"].update("Predicting...") # 4
    TaskOutput = USERINPUT_Model.predict(USERINPUT_TaskInput["F"])[0]
    # Display Outputs
    PROGRESS_BARS["overall"].update("Visualising Task Output...") # 5
    st.markdown("## Task Output")
    UI_DisplayLabelDistribution(TaskOutput, USERINPUT_Model.features_info["target"]["type"]["categories"])
    # Display Model Visualisations
    PROGRESS_BARS["overall"].update("Visualising Model...") # 6
    st.markdown("## Model Visualisation")
    # VisData = USERINPUT_Model.visualise()
    # UI_DisplayVisData(VisData)
    PROGRESS_BARS["overall"].update("Finished") # 7
    PROGRESS_BARS["overall"].finish()

# Mode Vars
APP_MODES = {
    "Text Problems - Sentiment Analysis": {
        "Pretrained": functools.partial(textproblems_init_basic, TASK="Sentiment Analysis"),
        "Train": functools.partial(textproblems_train_basic, TASK="Sentiment Analysis"),
        "Test": functools.partial(textproblems_test_basic, TASK="Sentiment Analysis")
    }
}

# App Functions
def app_main():
    # Title
    st.markdown("# MTech Project - Text Problems")
    # Mode
    USERINPUT_App = st.sidebar.selectbox(
        "Select App",
        list(APP_MODES.keys())
    )
    USERINPUT_Mode = st.sidebar.selectbox(
        "Select Mode",
        list(APP_MODES[USERINPUT_App].keys())
    )
    APP_MODES[USERINPUT_App][USERINPUT_Mode]()

def app_settings():
    global SETTINGS
    # Title
    st.markdown("# Settings")
    # Load Settings
    if SETTINGS["kaggle"]["username"] == "" or SETTINGS["kaggle"]["key"] == "":
        if os.path.exists(PATHS["credentials_kaggle"]): SETTINGS["kaggle"] = json.load(open(PATHS["credentials_kaggle"], "r"))
    # Settings
    SETTINGS["plots_interactive"] = st.checkbox("Interactive Plots", False)
    SETTINGS["kaggle"] = json.loads(st.text_area("Kaggle", json.dumps(SETTINGS["kaggle"], indent=4), height=250))
    # Save Settings
    if st.button("Save Settings"):
        json.dump(SETTINGS, open(PATHS["settings"], "w"), indent=4)
        # Settings Operations
        os.makedirs(os.path.dirname(PATHS["credentials_kaggle"]), exist_ok=True)
        if not (SETTINGS["kaggle"]["username"] == "" or SETTINGS["kaggle"]["key"] == ""):
            json.dump(SETTINGS["kaggle"], open(PATHS["credentials_kaggle"], "w"))
        st.success("Settings Saved")

# RunCode
if __name__ == "__main__":
    # Assign Objects
    SETTINGS = json.load(open(PATHS["settings"], "r"))
    SETTINGS_ACTIVE = st.sidebar.checkbox("Show Settings", False)
    if SETTINGS_ACTIVE:
        # Run Settings
        app_settings()
    else:
        # Run Main
        app_main()