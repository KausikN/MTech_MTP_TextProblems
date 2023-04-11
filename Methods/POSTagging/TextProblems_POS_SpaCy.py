"""
Text Problems - POS Tagging - SpaCy

References:

"""

# Imports
from Utils.Utils import *

# Main Functions
def Library_POS_SpaCy(text, **params):
    '''
    Library - POS Tagging - SpaCy
    '''
    # Run
    output = NLP(text)
    OUT = {
        "pos_tags": [
            {
                "token": str(d.text), 
                "pos_tag": str(d.pos_),
                "span": [d.i, d.i+1]
            }
            for d in output
        ],
        # "pos_obj": output
    }

    return OUT

# Main Vars
TASK_FUNCS = {
    "SpaCy": {
        "func": Library_POS_SpaCy,
        "params": {}
    }
}