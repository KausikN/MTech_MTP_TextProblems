"""
Text Problems - POS Tagging - NLTK

References:

"""

# Imports
from Utils.Utils import *

# Main Functions
def Library_POS_NLTK(text, **params):
    '''
    Library - POS Tagging - NLTK
    '''
    # Run
    output = nltk.pos_tag(nltk.word_tokenize(text), tagset="universal")
    OUT = {
        "pos_tags": []
    }
    curI = 0
    for di, d in enumerate(output):
        OUT["pos_tags"].append({
            "token": str(d[0]), 
            "pos_tag": str(d[1]),
            "span": [curI, curI+1]
        })
        curI += 1

    return OUT

# Main Vars
TASK_FUNCS = {
    "NLTK": {
        "func": Library_POS_NLTK,
        "params": {}
    }
}