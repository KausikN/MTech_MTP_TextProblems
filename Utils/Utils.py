"""
Utils
"""

# Imports
import os
import time
import json
## NLTK
import nltk
nltk.download("omw-1.4")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")
# SpaCy
import spacy
from spacy import displacy
from spacy.tokens.span import Span as spacy_span
NLP = spacy.load("en_core_web_sm")
# NetworkX
import networkx as nx

# TQDM
CONFIG = json.load(open(os.path.join(os.path.dirname(__file__), "..", "config.json"), "r"))
if CONFIG["tqdm_notebook"]:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# Util Functions
def name_to_path(name):
    # Convert to Lowercase
    name = name.lower()
    # Remove Special Chars
    for c in [" ", "-", ".", "<", ">", "/", "\\", ",", ";", ":", "'", '"', "|", "*", "?"]:
        name = name.replace(c, "_")

    return name

# Main Functions
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