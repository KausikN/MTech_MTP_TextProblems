"""
Text Problems - Summarisation - Gensim

References:

"""

# Imports
from Utils.Utils import *

from gensim.summarization.summarizer import summarize as GENSIM_SUMMARIZE

# Main Functions
def Library_Summarisation_Gensim(text, ratio=0.25, word_count=None, **params):
    '''
    Library - Summarisation - Gensim
    '''
    # Run
    output = GENSIM_SUMMARIZE(text, ratio=ratio, word_count=word_count)
    OUT = {
        "summary": output
    }

    return OUT

# Main Vars
TASK_FUNCS = {
    "Gensim": {
        "func": Library_Summarisation_Gensim,
        "params": {
            "ratio": 0.25,
            "word_count": None
        }
    }
}