"""
Text Problems - Sentiment Analysis - Pattern

References:

"""

# Imports
from Utils.Utils import *

import pattern.en as pattern_en

# Main Functions
def Library_SentimentAnalysis_Pattern(text, **params):
    '''
    Library - Sentiment Analysis - Pattern
    '''
    # Run
    output = pattern_en.sentiment(text)
    OUT = {
        "polarity": output[0],
        "subjectivity": output[1]
    }

    return OUT

# Main Vars
TASK_FUNCS = {
    "Pattern": {
        "func": Library_SentimentAnalysis_Pattern,
        "params": {}
    }
}