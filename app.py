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

# Cache Data Functions

# UI Functions
def UI_LoadTaskInput(TASK):
    '''
    Load Task Input
    '''
    # Init
    st.markdown("## Load Input")
    # Select Input Mode
    USERINPUT_InputMode = st.selectbox("Select Input Mode", ["Input"])
    if USERINPUT_InputMode == "Input":
        if TASK == "Sentiment Analysis":
            USERINPUT_Input = {
                "text": st.text_input("Enter Text")
            }
        elif TASK == "Named Entity Recognition":
            USERINPUT_Input = {
                "text": st.text_input("Enter Text")
            }
        elif TASK == "Relationship Extraction":
            USERINPUT_Input = {
                "text": st.text_input("Enter Text")
            }

    return USERINPUT_Input

def UI_SelectLibrary(TASK):
    '''
    UI - Select Library
    '''
    # Init
    st.markdown("## Select Library")
    # Select Library
    cols = st.columns((1, 3))
    USERINPUT_LibraryName = cols[0].selectbox("Select Library", list(TASK_MODULES[TASK].keys()))
    USERINPUT_LibraryParams = json.loads(cols[1].text_area(
        "Library Params", 
        value=json.dumps(TASK_MODULES[TASK][USERINPUT_LibraryName]["params"], indent=8),
        height=200
    ))
    # Select Library Method
    USERINPUT_Library = {
        "func": TASK_MODULES[TASK][USERINPUT_LibraryName]["func"],
        "params": USERINPUT_LibraryParams
    }

    return USERINPUT_Library

def UI_DisplayOutput(OUTPUT, USERINPUT_Input, TASK="Sentiment Analysis"):
    '''
    UI - Display Output
    '''
    # Init
    st.markdown("## Output")
    # JSON Output
    st.json(OUTPUT, expanded=False)
    # Task Specific Output
    if TASK == "Sentiment Analysis":
        ## Construct Plot
        fig = plt.figure()
        plt.bar(list(OUTPUT.keys()), OUTPUT.values())
        plt.title("Sentiment Analysis")
        ## Display
        PLOT_FUNC = st.plotly_chart if SETTINGS["plots_interactive"] else st.pyplot
        PLOT_FUNC(fig)
    elif TASK == "Named Entity Recognition":
        ## Construct Spacy Doc
        CurDoc = NLP(USERINPUT_Input["text"])
        Ents = []
        for i in range(len(OUTPUT["ner_tags"])):
            d = OUTPUT["ner_tags"][i]
            Ents.append(spacy_span(CurDoc, d["span"][0], d["span"][1], d["ner_tag"]))
        CurDoc.set_ents(Ents)
        ## Display
        RENDER_HTML = displacy.render(CurDoc, style="ent", minify=True)
        st.write(RENDER_HTML, unsafe_allow_html=True)
    elif TASK == "Relationship Extraction":
        ## Construct NetworkX Graph
        CurGraph = nx.DiGraph()
        for i in range(len(OUTPUT["relations"])):
            d = OUTPUT["relations"][i]
            CurGraph.add_edge(d["subject"], d["object"], relation=d["relation"])
        ## Construct Plot
        fig = plt.figure()
        LAYOUT = nx.spring_layout
        POS = LAYOUT(CurGraph)
        nx.draw_networkx(CurGraph, pos=POS, with_labels=True, arrows=True)
        nx.draw_networkx_edge_labels(CurGraph, pos=POS, edge_labels=nx.get_edge_attributes(CurGraph, "relation"))
        plt.title("Relationship Extraction")
        ## Display
        PLOT_FUNC = st.plotly_chart if SETTINGS["plots_interactive"] else st.pyplot
        PLOT_FUNC(fig)

# Main Functions
def textproblems_library(TASK="Sentiment Analysis"):
    # Title
    st.markdown(f"# Library - {TASK}")

    # Load Inputs
    # Init
    PROGRESS_BARS = {}
    # Load Input
    USERINPUT_Input = UI_LoadTaskInput(TASK)
    # Select Library
    USERINPUT_Library = UI_SelectLibrary(TASK)

    # Process Check
    USERINPUT_Process = st.checkbox("Stream Process", value=False)
    if not USERINPUT_Process: USERINPUT_Process = st.button("Process")
    if not USERINPUT_Process: st.stop()
    # Process Inputs
    ## Run Library
    OUTPUT = USERINPUT_Library["func"](
        USERINPUT_Input["text"],
        **USERINPUT_Library["params"]
    )
    ## Show Output
    UI_DisplayOutput(OUTPUT, USERINPUT_Input, TASK=TASK)

# Mode Vars
APP_MODES = {
    "Library": textproblems_library
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
    USERINPUT_Task = st.sidebar.selectbox(
        "Select Task",
        list(TASK_MODULES.keys())
    )
    APP_MODES[USERINPUT_App](USERINPUT_Task)

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