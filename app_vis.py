"""
Streamlit App - Visualisation
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
import plotly.express as px
import matplotlib.pyplot as plt
from evaluate.visualization import radar_plot

# from TextProblems import *
from Utils.Utils import *

# Main Vars
PATHS = {
    "temp": "Data/Temp/",
    "settings": "_appdata/settings.json",

    "data": {
        "evaluations": "Data/Evaluations/",
        "models": "Data/Models/",
    }
}
SETTINGS = {}
## Eval Vars
EVAL_METRICS_FILTER = {
    "Sentiment Analysis": {
        "rank_weights": {
            "accuracy": 1.0,
            "latency_in_seconds": 1.0
        }
    },
    "Named Entity Recognition": {
        "keep": [
            "overall_precision", "overall_recall", "overall_f1", "overall_accuracy", 
            "total_time_in_seconds", "samples_per_second", "latency_in_seconds", 
            "model_parameter_count"
        ],
        "rank_weights": {
            "overall_f1": 1.0,
            "latency_in_seconds": 1.0
        }
    },
    "POS Tagging": {
        "keep": [
            "overall_precision", "overall_recall", "overall_f1", "overall_accuracy", 
            "total_time_in_seconds", "samples_per_second", "latency_in_seconds", 
            "model_parameter_count"
        ],
        "rank_weights": {
            "overall_f1": 1.0,
            "latency_in_seconds": 1.0
        }
    },
    # "Relationship Extraction": {},
    # "Dialogue": {},
    "Summarisation": {
        "rank_weights": {
            "rouge1": 1.0,
            "latency_in_seconds": 1.0
        }
    },
    "Translation": {
        "ignore": [
            "bleu_precisions", "counts", "totals", "sacrebleu_precisions", 
            "translation_length", "reference_length", "bp", "sys_len", "ref_len"
        ],
        "rank_weights": {
            "bleu": 1.0,
            "latency_in_seconds": 1.0
        }
    },
    "Question Answering": {
        "ignore": [
            "total", "HasAns_total", "NoAns_total"
        ],
        "rank_weights": {
            "f1": 1.0,
            "latency_in_seconds": 1.0
        }
    }
}
METRICS_INVERT_RANGE = [
    "total_time_in_seconds", 
    "latency_in_seconds", 
    "model_parameter_count"
]
## Models Vars


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

# Cache Data Functions

# UI Functions
def UI_SelectEvalDataset(TASK):
    '''
    UI - Select Evaluated Dataset
    '''
    # Load All Eval Datasets
    EVAL_DATASETS = sorted(os.listdir(os.path.join(PATHS["data"]["evaluations"], name_to_path(TASK))))
    # Select Eval Dataset
    USERINPUT_EvalDataset = st.selectbox(
        "Select Evaluated Dataset",
        EVAL_DATASETS
    )

    return USERINPUT_EvalDataset

def UI_SelectEvalModels(TASK, USERINPUT_EvalDataset):
    '''
    UI - Select Evaluated Models
    '''
    # Load All Eval Models
    EVAL_MODELS = sorted(os.listdir(os.path.join(PATHS["data"]["evaluations"], name_to_path(TASK), USERINPUT_EvalDataset)))
    # Select Eval Models
    USERINPUT_EvalModels = st.multiselect(
        "Select Evaluated Models",
        EVAL_MODELS, 
        default=EVAL_MODELS
    )

    return USERINPUT_EvalModels

def UI_EvalVis(TASK, USERINPUT_EvalDataset, USERINPUT_EvalModels):
    '''
    UI - Visualise Evaluations
    '''
    # Init Plot Func
    PLOT_FUNC = st.plotly_chart if SETTINGS["plots_interactive"] else st.pyplot
    # Init
    DATASET_DIR = USERINPUT_EvalDataset
    EVAL_MODELS_NAMES = USERINPUT_EvalModels
    EVAL_FILE_NAME = "eval.json"
    EVAL_MODELS_PARENT_DIR = os.path.join(PATHS["data"]["evaluations"], name_to_path(TASK), USERINPUT_EvalDataset)
    OVERALL_RANK_METRIC_WEIGHTS = EVAL_METRICS_FILTER[TASK]["rank_weights"]
    # Load Eval Data
    MODELS_EVAL_DATA = [json.load(open(os.path.join(EVAL_MODELS_PARENT_DIR, f, EVAL_FILE_NAME), "r")) for f in EVAL_MODELS_NAMES]
    METRICS_DATA = [d["eval"] for d in MODELS_EVAL_DATA]
    # Filter Eval Metrics
    if "ignore" in EVAL_METRICS_FILTER[TASK].keys():
        METRICS_DATA = [{
            k: d[k] for k in d.keys() 
            if k not in EVAL_METRICS_FILTER[TASK]["ignore"]
        } for d in METRICS_DATA]
    if "keep" in EVAL_METRICS_FILTER[TASK].keys():
        METRICS_DATA = [{
            k: d[k] for k in d.keys() 
            if k in EVAL_METRICS_FILTER[TASK]["keep"]
        } for d in METRICS_DATA]
    EVAL_METRIC_KEYS = list(METRICS_DATA[0].keys())
    # Rank Models
    # Compute Rankings based on each metric
    RANKINGS = {}
    RANKING_NAMES = {}
    for em in EVAL_METRIC_KEYS:
        RANKINGS[em] = np.argsort([d[em] for d in METRICS_DATA])
        if em not in METRICS_INVERT_RANGE: RANKINGS[em] = RANKINGS[em][::-1]
        RANKING_NAMES[em] = [EVAL_MODELS_NAMES[r] for r in RANKINGS[em]]
    # Compute Overall Weighted Ranking
    OVERALL_RANKINGS = []
    OVERALL_RANKING_NAMES = []
    JOINT_RANK = [0] * len(EVAL_MODELS_NAMES)
    for em in OVERALL_RANK_METRIC_WEIGHTS.keys():
        for ri in range(len(RANKINGS[em])):
            JOINT_RANK[RANKINGS[em][ri]] += OVERALL_RANK_METRIC_WEIGHTS[em] * ri
    OVERALL_RANKINGS = np.argsort(JOINT_RANK)
    OVERALL_RANKING_NAMES = [EVAL_MODELS_NAMES[r] for r in OVERALL_RANKINGS]
    # Record
    RANK_DATA = {
        "dataset": DATASET_DIR,
        "ranking": RANKING_NAMES,
        "overall_ranking": {
            "params": {
                "weights": OVERALL_RANK_METRIC_WEIGHTS
            },
            "rank": OVERALL_RANKING_NAMES
        }
    }
    
    # Display Evals
    st.markdown("## Evaluations")
    # Display Evals Table
    EVALS_DF = pd.DataFrame(METRICS_DATA, index=EVAL_MODELS_NAMES)
    st.write(EVALS_DF)
    # Display Evals Plots
    # Plot Rank Data
    ############# Using Plotly
    ## Init
    N_MODELS = len(RANK_DATA["overall_ranking"]["rank"])
    SCORE_BAR_DICT = {}
    for em in OVERALL_RANK_METRIC_WEIGHTS.keys():
        SCORE_BAR_DICT[em] = [0] * N_MODELS
        for ri in range(len(RANKINGS[em])):
            SCORE_BAR_DICT[em][RANKINGS[em][ri]] = N_MODELS - ri
    RANK_BAR_DATA = pd.DataFrame({
        "model": EVAL_MODELS_NAMES,
        **SCORE_BAR_DICT
    })
    ## Plot
    FIG_RANK_BAR = px.bar(
        RANK_BAR_DATA, x="model", y=list(SCORE_BAR_DICT.keys()), 
        title="Overall Score: " + str(OVERALL_RANK_METRIC_WEIGHTS)
    )
    st.markdown("## Overall Ranking Score")
    PLOT_FUNC(FIG_RANK_BAR)
    # Plot Radar
    FIG_RADAR = plt.figure(figsize=(10, 10))
    FIG_RADAR = radar_plot(
        METRICS_DATA, 
        EVAL_MODELS_NAMES,
        invert_range=METRICS_INVERT_RANGE,
        fig=FIG_RADAR
    )
    st.markdown("## Radar Plot")
    st.pyplot(FIG_RADAR)

def UI_ModelsVis(MODELS_DATA, params):
    '''
    UI - Visualise Models
    '''
    # Init
    N_MODELS = len(MODELS_DATA)
    # Display Models
    st.markdown("## Models")
    # Display Models Table
    MODELS_DF_DATA = {
        "Reported Metrics": [],
        "Computed Metrics": []
    }
    MODELS_CHECK = {k: False for k in MODELS_DF_DATA.keys()}
    for m in MODELS_DATA:
        ## Basic Data
        for dfk in MODELS_DF_DATA.keys():
            MODELS_DF_DATA[dfk].append({
                "Model": m["Model"],
                "Code": m["Code"],
                "Language": m["Language"]
            })
        ## Metrics
        for mtk in ["Reported Metrics", "Computed Metrics"]:
            if mtk in m.keys():
                for dk in m[mtk].keys():
                    for emk in m[mtk][dk].keys():
                        MODELS_DF_DATA[mtk][-1][f"{emk} - {dk}"] = m[mtk][dk][emk]
                    if not MODELS_CHECK[mtk] and len(m[mtk][dk].keys()) > 0: MODELS_CHECK[mtk] = True

    for dfk in MODELS_DF_DATA.keys():
        if not MODELS_CHECK[dfk]: continue
        MODELS_DF = pd.DataFrame(MODELS_DF_DATA[dfk])
        st.markdown(f"### {dfk}")
        st.write(MODELS_DF)

# Main Functions
def textproblems_vis_evals_basic(TASK="Sentiment Analysis"):
    # Title
    st.markdown(f"# Evaluations - {TASK}")

    # Load Inputs
    # Init
    PROGRESS_BARS = {}
    # Select Dataset
    USERINPUT_EvalDataset = UI_SelectEvalDataset(TASK)
    # Select Models
    USERINPUT_EvalModels = UI_SelectEvalModels(TASK, USERINPUT_EvalDataset)

    # Process Check
    USERINPUT_Process = st.checkbox("Stream Process", value=False)
    if not USERINPUT_Process: USERINPUT_Process = st.button("Process")
    if not USERINPUT_Process: st.stop()
    # Process Inputs
    ## Run Vis
    UI_EvalVis(TASK, USERINPUT_EvalDataset, USERINPUT_EvalModels)

def textproblems_vis_models_basic(TASK="Sentiment Analysis"):
    # Title
    st.markdown(f"# Models - {TASK}")

    # Load Inputs
    # Init
    PROGRESS_BARS = {}
    # Load Models for Task
    TASK_MODELS_DATA = json.load(open(os.path.join(
        PATHS["data"]["models"], TASK, f"MTP - Text Problems - {TASK} - Models.json"
    ), "r"))
    # Select Task Subtype
    USERINPUT_TaskSubtype = st.selectbox(
        "Select Task Subtype",
        TASK_MODELS_DATA.keys()
    )
    # Select Model Type
    USERINPUT_ModelType = st.selectbox(
        "Select Model Type",
        TASK_MODELS_DATA[USERINPUT_TaskSubtype].keys()
    )
    # Select Model Subtype
    USERINPUT_ModelSubtype = st.selectbox(
        "Select Model Subtype",
        TASK_MODELS_DATA[USERINPUT_TaskSubtype][USERINPUT_ModelType].keys()
    )
    # Select Dataset
    USERINPUT_Dataset = st.selectbox(
        "Select Dataset",
        TASK_MODELS_DATA[USERINPUT_TaskSubtype][USERINPUT_ModelType][USERINPUT_ModelSubtype].keys()
    )

    # Process Check
    USERINPUT_Process = st.checkbox("Stream Process", value=False)
    if not USERINPUT_Process: USERINPUT_Process = st.button("Process")
    if not USERINPUT_Process: st.stop()
    # Process Inputs
    ## Load Data
    MODELS_DATA = TASK_MODELS_DATA[USERINPUT_TaskSubtype][USERINPUT_ModelType][USERINPUT_ModelSubtype][USERINPUT_Dataset]
    ## Run Vis
    USERINPUT_Selections = {
        "task": TASK,
        "task_subtype": USERINPUT_TaskSubtype,
        "model_type": USERINPUT_ModelType,
        "model_subtype": USERINPUT_ModelSubtype,
        "dataset": USERINPUT_Dataset
    }
    UI_ModelsVis(MODELS_DATA, USERINPUT_Selections)

# Mode Vars
APP_MODES = {
    "Evaluations": {
        k: functools.partial(textproblems_vis_evals_basic, TASK=k) for k in [
            "Sentiment Analysis",
            "Named Entity Recognition",
            "POS Tagging",
            # "Relationship Extraction",
            # "Dialogue",
            "Summarisation",
            "Translation",
            "Question Answering"
        ]
    },
    "Models": {
        k: functools.partial(textproblems_vis_models_basic, TASK=k) for k in [
            "Sentiment Analysis",
            "Named Entity Recognition",
            # "POS Tagging",
            "Relationship Extraction",
            "Dialogue",
            "Summarisation",
            "Translation",
            "Question Answering"
        ]
    }
}

# App Functions
def app_main():
    # Title
    st.markdown("# MTech Project - Text Problems - Visualisation")
    os.makedirs(PATHS["temp"], exist_ok=True)
    # Mode
    USERINPUT_App = st.sidebar.selectbox(
        "Select App",
        list(APP_MODES.keys())
    )
    USERINPUT_Task = st.sidebar.selectbox(
        "Select Task",
        list(APP_MODES[USERINPUT_App].keys())
    )
    APP_MODES[USERINPUT_App][USERINPUT_Task]()

def app_settings():
    global SETTINGS
    # Title
    st.markdown("# Settings")
    # Settings
    SETTINGS["plots_interactive"] = st.checkbox("Interactive Plots", False)
    # Save Settings
    if st.button("Save Settings"):
        json.dump(SETTINGS, open(PATHS["settings"], "w"), indent=4)
        st.success("Settings Saved")

# RunCode
if __name__ == "__main__":
    # Assign Objects
    SETTINGS = json.load(open(PATHS["settings"], "r"))

    SETTINGS["plots_interactive"] = st.sidebar.checkbox("Interactive Plots", True)
    # SETTINGS_ACTIVE = st.sidebar.checkbox("Show Settings", False)
    # if SETTINGS_ACTIVE:
    #     # Run Settings
    #     app_settings()
    # else:
    #     # Run Main
    #     app_main()

    app_main()