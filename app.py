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
    "settings": "_appdata/settings.json",

    "cache": {
        "Dialogue": {
            "history": "Data/Temp/Dialogue_History.json",
            "objects": "Data/Temp/Dialogue_Objects.p"
        }
    }
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
        "name": USERINPUT_LibraryName,
        "func": TASK_MODULES[TASK][USERINPUT_LibraryName]["func"],
        "params": USERINPUT_LibraryParams
    }

    return USERINPUT_Library

def UI_LoadTaskInput(TASK, LibraryName="SpaCy"):
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
                "text": st.text_input("Enter Text", value=EXAMPLE_TEXTS[TASK])
            }
        elif TASK == "Named Entity Recognition":
            USERINPUT_Input = {
                "text": st.text_input("Enter Text", value=EXAMPLE_TEXTS[TASK])
            }
        elif TASK == "Relationship Extraction":
            USERINPUT_Input = {
                "text": st.text_input("Enter Text", value=EXAMPLE_TEXTS[TASK])
            }
        elif TASK == "Summarisation":
            USERINPUT_Input = {
                "text": st.text_area("Enter Text", value=EXAMPLE_TEXTS[TASK], height=200)
            }
        elif TASK == "Translation":
            USERINPUT_Input = {
                "text": st.text_area("Enter Text", value=EXAMPLE_TEXTS[TASK], height=200)
            }

    return USERINPUT_Input

def UI_DisplayOutput(OUTPUT, USERINPUT_Input, TASK="Sentiment Analysis", LibraryName="SpaCy"):
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
    elif TASK == "Dialogue":
        ## Display
        st.markdown("## Bot Reply")
        st.markdown("USER: " + USERINPUT_Input["text"])
        st.markdown("BOT: " + OUTPUT["response"])
    elif TASK == "Summarisation":
        ## Display
        st.markdown("## Summary")
        st.text_area("Summary", value=OUTPUT["summary"], height=200, disabled=True)
    elif TASK == "Translation":
        ## Display
        st.markdown("## Translated Text")
        st.text_area("Translated Text", value=OUTPUT["translated_text"], height=200, disabled=True)

# Main Functions
def textproblems_library_basic(TASK="Sentiment Analysis"):
    # Title
    st.markdown(f"# Library - {TASK}")

    # Load Inputs
    # Init
    PROGRESS_BARS = {}
    # Select Library
    USERINPUT_Library = UI_SelectLibrary(TASK)
    # Load Input
    USERINPUT_Input = UI_LoadTaskInput(TASK, USERINPUT_Library["name"])

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
    UI_DisplayOutput(OUTPUT, USERINPUT_Input, TASK=TASK, LibraryName=USERINPUT_Library["name"])

def textproblems_library_dialogue():
    # Title
    TASK = "Dialogue"
    st.markdown(f"# Library - {TASK}")

    # Load Inputs
    # Init
    PROGRESS_BARS = {}
    # Select Library
    USERINPUT_Library = UI_SelectLibrary(TASK)
    # Load Input
    CACHE_PATH = PATHS["cache"]["Dialogue"]
    ## Load Cache
    CACHE_DATA = {
        "history": [] if not os.path.exists(CACHE_PATH["history"]) else json.load(open(CACHE_PATH["history"], "r"))["history"],
        "objects": None if not os.path.exists(CACHE_PATH["objects"]) else pickle.load(open(CACHE_PATH["objects"], "rb"))
    }
    ## Load Objects
    if CACHE_DATA["objects"] is not None:
        TASK_OBJECTS[TASK][USERINPUT_Library["name"]]["module"].TASK_OBJECTS.update(CACHE_DATA["objects"])
    else:
        LOAD_FUNCS = TASK_OBJECTS[TASK][USERINPUT_Library["name"]]["module"].TASK_OBJECTS_LOAD_FUNCS
        for k in LOAD_FUNCS.keys(): LOAD_FUNCS[k]()
    ## Display History
    st.markdown("## History")
    history_lines = []
    for i in range(len(CACHE_DATA["history"])):
        user_str = "USER: " if (i%2==0) else "BOT: "
        history_lines.append(user_str + CACHE_DATA["history"][i])
    st.markdown("```" + "\n".join(history_lines) + "```")
    ### Clear History
    if st.button("Clear History"):
        CACHE_DATA["history"] = []
        CACHE_DATA["objects"] = None
        if os.path.exists(CACHE_PATH["history"]): os.remove(CACHE_PATH["history"])
        if os.path.exists(CACHE_PATH["objects"]): os.remove(CACHE_PATH["objects"])
    ## Load Input
    USERINPUT_Input = {
        "text": st.text_input("Enter Text", value=EXAMPLE_TEXTS[TASK]),
        "history": CACHE_DATA["history"]
    }

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
    ## Update History
    USERINPUT_Input["history"].append(USERINPUT_Input["text"])
    USERINPUT_Input["history"].append(OUTPUT["reply"])
    ## Save Cache
    CACHE_DATA = {
        "history": USERINPUT_Input["history"],
        "objects": TASK_OBJECTS[TASK][USERINPUT_Library["name"]]["module"].TASK_OBJECTS
    }
    json.dump({"history": CACHE_DATA["history"]}, open(CACHE_PATH["history"], "w"), indent=4)
    # pickle.dump(CACHE_DATA["objects"], open(CACHE_PATH, "wb"))
    ## Show Output
    UI_DisplayOutput(OUTPUT, USERINPUT_Input, TASK=TASK, LibraryName=USERINPUT_Library["name"])

# Mode Vars
APP_MODES = {
    "Library": {
        "Sentiment Analysis": functools.partial(textproblems_library_basic, TASK="Sentiment Analysis"),
        "Named Entity Recognition": functools.partial(textproblems_library_basic, TASK="Named Entity Recognition"),
        "Relationship Extraction": functools.partial(textproblems_library_basic, TASK="Relationship Extraction"),
        "Dialogue": textproblems_library_dialogue,
        "Summarisation": functools.partial(textproblems_library_basic, TASK="Summarisation"),
        "Translation": functools.partial(textproblems_library_basic, TASK="Translation")
    }
}

# App Functions
def app_main():
    # Title
    st.markdown("# MTech Project - Text Problems")
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