"""
Text Problems - Sentiment Analysis - TextBlob

References:

"""

# Imports
from Utils.Utils import *

from textblob import TextBlob

# Main Functions
def Library_SentimentAnalysis_TextBlob(text, **params):
    '''
    Library - Sentiment Analysis - TextBlob
    '''
    # Run
    output = TextBlob(text).sentiment
    OUT = {
        "polarity": output.polarity,
        "subjectivity": output.subjectivity,
    }

    return OUT

# Main Vars
TASK_FUNCS = {
    "TextBlob": {
        "func": Library_SentimentAnalysis_TextBlob,
        "params": {}
    }
}