"""
Text Problems - Named Entity Recognition - NLTK

References:

"""

# Imports
from Utils.Utils import *

# Main Functions
def Library_NER_NLTK(text, **params):
    '''
    Library - Named Entity Recognition - NLTK
    '''
    # Run
    output = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
    OUT = {
        "ner_tags": []
    }
    curI = 0
    for di, d in enumerate(output):
        if hasattr(d, "label"):
            OUT["ner_tags"].append({
                "token": " ".join([str(w[0]) for w in d]), 
                "ner_tag": str(d.label()),
                "span": [curI, curI+len(d)]
            })
            curI += len(d)
        else:
            curI += 1

    return OUT

# Main Vars
TASK_FUNCS = {
    "NLTK": {
        "func": Library_NER_NLTK,
        "params": {}
    }
}