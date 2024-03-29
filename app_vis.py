"""
Streamlit App - Visualisation

Total Models in Models Data: 454
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
from sklearn.tree import DecisionTreeClassifier, plot_tree

# from TextProblems import *
from Utils.Utils import *

# Main Vars
PATHS = {
    "temp": "Data/Temp/",
    "settings": "_appdata/settings.json",

    "data": {
        "evaluations": "Data/Evaluations/",
        "evaluations_file": "eval.json",
        "models": "Data/Models/",
        "models_file": "MTP - Text Problems - {TASK} - Models.json"
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
def Utils_GetModelsInfo(TASK_MODELS_DATA):
    '''
    Utils - Get Task Models Info
    '''
    # Init
    TASK_MODELS_INFO = {
        "overall": {},
        "separate": {}
    }
    # Iterate over subtasks
    for subtask in TASK_MODELS_DATA.keys():
        ## Init
        TASK_MODELS_INFO["separate"][subtask] = {
            "overall": {},
            "separate": {}
        }
        CUR_INFO_SUBTASK = TASK_MODELS_INFO["separate"][subtask]
        CUR_MODELS_SUBTASK = TASK_MODELS_DATA[subtask]
        ## Iterate over model types
        for model_type in CUR_MODELS_SUBTASK.keys():
            ### Init
            CUR_INFO_SUBTASK["separate"][model_type] = {
                "overall": {},
                "separate": {}
            }
            CUR_INFO_MODELTYPE = CUR_INFO_SUBTASK["separate"][model_type]
            CUR_MODELS_MODELTYPE = CUR_MODELS_SUBTASK[model_type]
            ### Iterate over models subtypes
            for model_subtype in CUR_MODELS_MODELTYPE.keys():
                #### Init
                CUR_INFO_MODELTYPE["separate"][model_subtype] = {
                    "overall": {},
                    "separate": {}
                }
                CUR_INFO_MODELSUBTYPE = CUR_INFO_MODELTYPE["separate"][model_subtype]
                CUR_MODELS_MODELSUBTYPE = CUR_MODELS_MODELTYPE[model_subtype]
                #### Iterate over datasets
                for dataset in CUR_MODELS_MODELSUBTYPE.keys():
                    ##### Init
                    CUR_INFO_MODELSUBTYPE["separate"][dataset] = {}
                    CUR_INFO = CUR_INFO_MODELSUBTYPE["separate"][dataset]
                    CUR_MODELS = CUR_MODELS_MODELSUBTYPE[dataset]
                    ##### Iterate over models
                    CUR_INFO.update({
                        "models_count": len(CUR_MODELS)
                    })
                #### Set Overall
                CUR_INFO_MODELSUBTYPE["overall"].update({
                    "models_count": sum([
                        CUR_INFO_MODELSUBTYPE["separate"][k]["models_count"] 
                        for k in CUR_INFO_MODELSUBTYPE["separate"].keys()
                    ])
                })
            ### Set Overall
            CUR_INFO_MODELTYPE["overall"].update({
                "models_count": sum([
                    CUR_INFO_MODELTYPE["separate"][k]["overall"]["models_count"] 
                    for k in CUR_INFO_MODELTYPE["separate"].keys()
                ])
            })
        ## Set Overall
        CUR_INFO_SUBTASK["overall"].update({
            "models_count": sum([
                CUR_INFO_SUBTASK["separate"][k]["overall"]["models_count"] 
                for k in CUR_INFO_SUBTASK["separate"].keys()
            ])
        })
    # Set Overall
    TASK_MODELS_INFO["overall"].update({
        "models_count": sum([
            TASK_MODELS_INFO["separate"][k]["overall"]["models_count"] 
            for k in TASK_MODELS_INFO["separate"].keys()
        ])
    })

    return TASK_MODELS_INFO

def Utils_PieChart(data, labels, title=""):
    '''
    Utils - Construct Pie Chart
    '''
    # Init
    CUR_FIG = plt.figure()
    labels_withvals = [f"{labels[i]} ({data[i]})" for i in range(len(data))]
    # Plot
    plt.pie(data, labels=labels_withvals, shadow=True, autopct="%1.2f%%", explode=[0.1]*len(data))
    plt.title(title)
    # plt.legend()

    return CUR_FIG

def Utils_GetModelsInfoVis(TASK_MODELS_INFO):
    '''
    Utils - Get Task Models Info Visualisations
    '''
    # Init
    TASK_MODELS_INFO_VIS = {
        "overall": {},
        "separate": {}
    }
    # Iterate over subtasks
    for subtask in TASK_MODELS_INFO["separate"].keys():
        ## Init
        TASK_MODELS_INFO_VIS["separate"][subtask] = {
            "overall": {},
            "separate": {}
        }
        CUR_VIS_SUBTASK = TASK_MODELS_INFO_VIS["separate"][subtask]
        CUR_INFO_SUBTASK = TASK_MODELS_INFO["separate"][subtask]
        ## Iterate over model types
        for model_type in CUR_INFO_SUBTASK["separate"].keys():
            ### Init
            CUR_VIS_SUBTASK["separate"][model_type] = {
                "overall": {},
                "separate": {}
            }
            CUR_VIS_MODELTYPE = CUR_VIS_SUBTASK["separate"][model_type]
            CUR_INFO_MODELTYPE = CUR_INFO_SUBTASK["separate"][model_type]
            ### Iterate over models subtypes
            for model_subtype in CUR_INFO_MODELTYPE["separate"].keys():
                #### Init
                CUR_VIS_MODELTYPE["separate"][model_subtype] = {
                    "overall": {},
                    "separate": {}
                }
                CUR_VIS_MODELSUBTYPE = CUR_VIS_MODELTYPE["separate"][model_subtype]
                CUR_INFO_MODELSUBTYPE = CUR_INFO_MODELTYPE["separate"][model_subtype]
                #### Set Overall
                CUR_VIS_MODELSUBTYPE["overall"].update({
                    "pie": Utils_PieChart(
                        [
                            CUR_INFO_MODELSUBTYPE["separate"][k]["models_count"] 
                            for k in CUR_INFO_MODELSUBTYPE["separate"].keys()
                        ],
                        labels=list(CUR_INFO_MODELSUBTYPE["separate"].keys()),
                        title=f"{subtask} - {model_type} - {model_subtype}"
                    )
                })
            ### Set Overall
            CUR_VIS_MODELTYPE["overall"].update({
                "pie": Utils_PieChart(
                    [
                        CUR_INFO_MODELTYPE["separate"][k]["overall"]["models_count"] 
                        for k in CUR_INFO_MODELTYPE["separate"].keys()
                    ],
                    labels=list(CUR_INFO_MODELTYPE["separate"].keys()),
                    title=f"{subtask} - {model_type}"
                )
            })
        ## Set Overall
        CUR_VIS_SUBTASK["overall"].update({
            "pie": Utils_PieChart(
                [
                    CUR_INFO_SUBTASK["separate"][k]["overall"]["models_count"] 
                    for k in CUR_INFO_SUBTASK["separate"].keys()
                ],
                labels=list(CUR_INFO_SUBTASK["separate"].keys()),
                title=f"{subtask}"
            )
        })
    # Set Overall
    TASK_MODELS_INFO_VIS["overall"].update({
        "pie": Utils_PieChart(
            [
                TASK_MODELS_INFO["separate"][k]["overall"]["models_count"] 
                for k in TASK_MODELS_INFO["separate"].keys()
            ],
            labels=list(TASK_MODELS_INFO["separate"].keys()),
            title=f"Overall"
        )
    })

    return TASK_MODELS_INFO_VIS

@st.cache_resource
def Utils_TrainEvalDecisionTree(METRICS_DATA):
    '''
    Utils - Train Decision Tree for Evaluations
    '''
    # Init
    EVAL_METRIC_KEYS = list(METRICS_DATA[0].keys())
    # Train Decision Tree
    F_keys = EVAL_METRIC_KEYS
    Fs = np.array([d[f] for d in METRICS_DATA for f in F_keys]).reshape(-1, len(F_keys))
    Ls = np.arange(len(METRICS_DATA)).reshape(-1)
    MAX_DEPTH = int(Ls.shape[0]/2)
    DECISION_TREE = DecisionTreeClassifier(
        criterion="entropy",
        random_state=0,
        max_depth=MAX_DEPTH
    ).fit(Fs, Ls)
    
    return DECISION_TREE

def Utils_LoadEvalMetrics(TASK, EVAL_MODELS_NAMES, EVAL_MODELS_PARENT_DIR, EVAL_FILE_NAME=PATHS["data"]["evaluations_file"]):
    '''
    Utils - Load Eval Metrics
    '''
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

    OUT = {
        "metrics_data": METRICS_DATA,
        "metric_keys": EVAL_METRIC_KEYS
    }
    return OUT

def Utils_GetModelIDMap(TASK):
    '''
    Utils - Get Model ID Map
    '''
    # Init
    ParentTask = TASK if name_to_path(TASK) != "pos_tagging" else "Named Entity Recognition"
    select_sub_task = lambda data: data[0] if name_to_path(TASK) != "pos_tagging" else data[1]
    # Load Models Data
    TASK_MODELS_DATA = json.load(open(os.path.join(
        PATHS["data"]["models"], ParentTask, PATHS["data"]["models_file"].format(TASK=ParentTask)
    ), "r"))
    TASK_MODELS_DATA = TASK_MODELS_DATA[select_sub_task(list(TASK_MODELS_DATA.keys()))]["Hugging Face"]
    MODELS_ID_MAP = {}
    for language in TASK_MODELS_DATA.keys():
        for dataset in TASK_MODELS_DATA[language].keys():
            for model_data in TASK_MODELS_DATA[language][dataset]:
                MODELS_ID_MAP[model_data["Eval"]] = {
                    "Hugging Face ID": model_data["Hugging Face ID"],
                    "Hugging Face Link": model_data["Code"],
                    "Base Model": model_data["Model"],
                    "Language": model_data["Language"],
                    "Reported Metrics": model_data["Reported Metrics"]
                }

    return MODELS_ID_MAP

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

def UI_DisplayDecisionTree(DECISION_TREE, FEATURE_NAMES, CLASS_NAMES, display=True):
    '''
    UI - Display Decision Tree
    '''
    # Plot Tree
    FIG_TREE = plt.figure(figsize=(50, 30))
    plot_tree(
        DECISION_TREE, 
        max_depth=None, 
        feature_names=FEATURE_NAMES, 
        class_names=CLASS_NAMES,
        label="all",
        impurity=False,
        filled=True,
        rounded=True,
        proportion=False
    )
    # st.pyplot(FIG_TREE)
    TREE_SAVE_PATH = os.path.join(PATHS["temp"], "tree.svg")
    FIG_TREE.savefig(TREE_SAVE_PATH)
    st.download_button(
        "Download Decision Tree",
        data=open(TREE_SAVE_PATH, "rb").read(),
        file_name="tree.svg",
        mime="image/svg+xml"
    )
    if display: st.image(TREE_SAVE_PATH, width=100)

def UI_DisplayMetricsDistribution(METRICS_DATA, n_cols=2):
    '''
    UI - Display Metrics Distribution
    '''
    # Init
    N_METRICS = len(METRICS_DATA[0].keys())
    N_ROWS = int(np.ceil(N_METRICS / n_cols))
    EVAL_METRICS = list(METRICS_DATA[0].keys())
    # Get Bins
    BINS = st.number_input("Bins", min_value=1, max_value=len(METRICS_DATA), value=min(5, len(METRICS_DATA)))
    # Display Metrics Distribution
    for row_i in range(N_ROWS):
        cols = st.columns(n_cols)
        for col_i in range(n_cols):
            i = row_i * n_cols + col_i
            if i >= N_METRICS: break
            FIG = plt.figure()
            METRIC_DATA = np.round([d[EVAL_METRICS[i]] for d in METRICS_DATA], 5)
            plt.hist(METRIC_DATA, bins=BINS)
            plt.title(list(METRICS_DATA[0].keys())[i])
            if SETTINGS["plots_interactive"]:
                cols[col_i].plotly_chart(FIG)
            else:
                cols[col_i].pyplot(FIG)

def UI_EvalVis(TASK, USERINPUT_EvalDataset, USERINPUT_EvalModels):
    '''
    UI - Visualise Evaluations
    '''
    # Init Plot Func
    PLOT_FUNC = st.plotly_chart if SETTINGS["plots_interactive"] else st.pyplot
    # Init
    DATASET_DIR = USERINPUT_EvalDataset
    EVAL_MODELS_NAMES = USERINPUT_EvalModels
    EVAL_MODELS_PARENT_DIR = os.path.join(PATHS["data"]["evaluations"], name_to_path(TASK), USERINPUT_EvalDataset)
    OVERALL_RANK_METRIC_WEIGHTS = EVAL_METRICS_FILTER[TASK]["rank_weights"]
    # Load Eval Data
    OUT = Utils_LoadEvalMetrics(TASK, EVAL_MODELS_NAMES, EVAL_MODELS_PARENT_DIR, EVAL_FILE_NAME=PATHS["data"]["evaluations_file"])
    METRICS_DATA = OUT["metrics_data"]
    EVAL_METRIC_KEYS = OUT["metric_keys"]
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

    # Construct Decision Tree
    st.markdown("## Decision Tree")
    DECISION_TREE = Utils_TrainEvalDecisionTree(METRICS_DATA)
    # Plot Tree
    UI_DisplayDecisionTree(DECISION_TREE, EVAL_METRIC_KEYS, EVAL_MODELS_NAMES)

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
def textproblems_vis_evals_decisiontree(TASK="Sentiment Analysis"):
    # Title
    st.markdown(f"# Recommend HF Model - {TASK}")

    # Load Inputs
    # Init
    PROGRESS_BARS = {}
    # Select Dataset
    USERINPUT_EvalDataset = UI_SelectEvalDataset(TASK)
    # Select Models
    USERINPUT_EvalModels = UI_SelectEvalModels(TASK, USERINPUT_EvalDataset)
    st.markdown(f"{len(USERINPUT_EvalModels)} Models Selected.")

    # Process Check
    USERINPUT_Process = st.checkbox("Load Metrics", value=False)
    if not USERINPUT_Process: st.stop()
    # Load Metrics
    EVAL_MODELS_NAMES = USERINPUT_EvalModels
    EVAL_MODELS_PARENT_DIR = os.path.join(PATHS["data"]["evaluations"], name_to_path(TASK), USERINPUT_EvalDataset)
    EVAL_FILE_NAME = PATHS["data"]["evaluations_file"]
    OUT_METRICS = Utils_LoadEvalMetrics(TASK, EVAL_MODELS_NAMES, EVAL_MODELS_PARENT_DIR, EVAL_FILE_NAME=EVAL_FILE_NAME)
    METRICS_DATA = OUT_METRICS["metrics_data"]
    EVAL_METRIC_KEYS = OUT_METRICS["metric_keys"]
    # Load Model ID Map
    MODELS_ID_MAP = Utils_GetModelIDMap(TASK)
    ## Select Metrics
    USERINPUT_Metrics = st.multiselect(
        "Select Metrics", EVAL_METRIC_KEYS, 
        default=EVAL_METRICS_FILTER[TASK]["rank_weights"].keys()
    )
    METRICS_DATA_REDUCED = [{
        k: d[k] for k in d.keys() 
        if k in USERINPUT_Metrics
    } for d in METRICS_DATA]
    EVAL_METRIC_KEYS_REDUCED = list(METRICS_DATA_REDUCED[0].keys())
    ## Display Metrics
    st.markdown("## Metrics")
    cols = st.columns(2)
    if cols[0].checkbox("Display Metrics", value=False):
        METRICS_DF = pd.DataFrame(METRICS_DATA_REDUCED, index=EVAL_MODELS_NAMES)
        st.write(METRICS_DF)
    if cols[1].checkbox("Display Metric Distributions", value=True):
        UI_DisplayMetricsDistribution(METRICS_DATA_REDUCED, n_cols=2)

    # Process Check
    USERINPUT_Process = st.checkbox("Process Decision Tree", value=False)
    if not USERINPUT_Process: st.stop()
    # Process Inputs
    ## Construct Decision Tree
    DECISION_TREE = Utils_TrainEvalDecisionTree(METRICS_DATA_REDUCED)
    # Plot Tree
    UI_DisplayDecisionTree(DECISION_TREE, EVAL_METRIC_KEYS_REDUCED, EVAL_MODELS_NAMES, display=False)

    # Process Check
    USERINPUT_Process = st.checkbox("Recommend Model", value=False)
    if not USERINPUT_Process: st.stop()
    ## Get Input Metrics
    USERINPUT_InputMetrics = json.loads(st.text_area(
        "Target Metrics", 
        value=json.dumps(METRICS_DATA_REDUCED[0], indent=8),
        height=200
    ))
    ## Predict Model
    PREDICTED_MODEL_INDEX = DECISION_TREE.predict([[
        USERINPUT_InputMetrics[k] for k in EVAL_METRIC_KEYS_REDUCED
    ]])[0]
    PREDICTED_MODEL_NAME = EVAL_MODELS_NAMES[PREDICTED_MODEL_INDEX]
    ## Display Predicted Model
    st.markdown("## Recommended Model")
    ### Display Model Info
    if PREDICTED_MODEL_NAME in MODELS_ID_MAP.keys():
        st.markdown(f"### Model Info")
        st.write(MODELS_ID_MAP[PREDICTED_MODEL_NAME])
    ### Display Eval Info
    PREDICTED_MODEL_DATA = json.load(open(os.path.join(EVAL_MODELS_PARENT_DIR, PREDICTED_MODEL_NAME, EVAL_FILE_NAME), "r"))
    st.markdown(f"### Evaluation Info")
    st.write(PREDICTED_MODEL_DATA)
    
    

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
    INFO_DISPLAY_COLRATIO = (1, 3)
    # Load Models for Task
    TASK_MODELS_DATA = json.load(open(os.path.join(
        PATHS["data"]["models"], TASK, PATHS["data"]["models_file"].format(TASK=TASK)
    ), "r"))
    # Get Models Info and InfoVis
    TASK_MODELS_INFO = Utils_GetModelsInfo(TASK_MODELS_DATA)
    TASK_MODELS_INFO_VIS = Utils_GetModelsInfoVis(TASK_MODELS_INFO)
    CUR_DATA = {
        "info": TASK_MODELS_INFO,
        "vis": TASK_MODELS_INFO_VIS
    }
    cols = st.columns(INFO_DISPLAY_COLRATIO)
    cols[0].markdown("Total Model Count: " + "```" + str(CUR_DATA["info"]["overall"]["models_count"]) + "```")
    cols[1].pyplot(CUR_DATA["vis"]["overall"]["pie"])
    # Select Task Subtype
    USERINPUT_TaskSubtype = st.selectbox(
        "Select Task Subtype",
        TASK_MODELS_DATA.keys()
    )
    CUR_DATA = {
        "info": CUR_DATA["info"]["separate"][USERINPUT_TaskSubtype],
        "vis": CUR_DATA["vis"]["separate"][USERINPUT_TaskSubtype]
    }
    cols = st.columns(INFO_DISPLAY_COLRATIO)
    cols[0].markdown(f"{USERINPUT_TaskSubtype} Model Count: " + "```" + str(CUR_DATA["info"]["overall"]["models_count"]) + "```")
    cols[1].pyplot(CUR_DATA["vis"]["overall"]["pie"])
    # Select Model Type
    USERINPUT_ModelType = st.selectbox(
        "Select Model Type",
        TASK_MODELS_DATA[USERINPUT_TaskSubtype].keys()
    )
    CUR_DATA = {
        "info": CUR_DATA["info"]["separate"][USERINPUT_ModelType],
        "vis": CUR_DATA["vis"]["separate"][USERINPUT_ModelType]
    }
    cols = st.columns(INFO_DISPLAY_COLRATIO)
    cols[0].markdown(f"{USERINPUT_ModelType} Model Count: " + "```" + str(CUR_DATA["info"]["overall"]["models_count"]) + "```")
    cols[1].pyplot(CUR_DATA["vis"]["overall"]["pie"])
    # Select Model Subtype
    USERINPUT_ModelSubtype = st.selectbox(
        "Select Model Subtype",
        TASK_MODELS_DATA[USERINPUT_TaskSubtype][USERINPUT_ModelType].keys()
    )
    CUR_DATA = {
        "info": CUR_DATA["info"]["separate"][USERINPUT_ModelSubtype],
        "vis": CUR_DATA["vis"]["separate"][USERINPUT_ModelSubtype]
    }
    cols = st.columns(INFO_DISPLAY_COLRATIO)
    cols[0].markdown(f"{USERINPUT_ModelSubtype} Model Count: " + "```" + str(CUR_DATA["info"]["overall"]["models_count"]) + "```")
    cols[1].pyplot(CUR_DATA["vis"]["overall"]["pie"])
    # Select Dataset
    USERINPUT_Dataset = st.selectbox(
        "Select Dataset",
        TASK_MODELS_DATA[USERINPUT_TaskSubtype][USERINPUT_ModelType][USERINPUT_ModelSubtype].keys()
    )
    CUR_DATA = {
        "info": CUR_DATA["info"]["separate"][USERINPUT_Dataset]
    }
    st.markdown(f"{USERINPUT_Dataset} Model Count: " + "```" + str(CUR_DATA["info"]["models_count"]) + "```")

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
    },
    "Recommend HF Model": {
        k: functools.partial(textproblems_vis_evals_decisiontree, TASK=k) for k in [
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