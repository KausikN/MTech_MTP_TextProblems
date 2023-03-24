"""
Text Problems - Named Entity Recognition - SpaCy

References:

"""

# Imports
from Utils.Utils import *

# Main Functions
def Library_NER_SpaCy(text, **params):
    '''
    Library - Named Entity Recognition - SpaCy
    '''
    # Run
    output = NLP(text)
    OUT = {
        "ner_tags": [
            {
                "token": str(d.text), 
                "ner_tag": str(d.label_),
                "span": [d.start, d.end]
            }
            for d in output.ents
        ],
        # "ner_obj": output
    }

    return OUT

# Main Vars
TASK_FUNCS = {
    "SpaCy": {
        "func": Library_NER_SpaCy,
        "params": {}
    }
}