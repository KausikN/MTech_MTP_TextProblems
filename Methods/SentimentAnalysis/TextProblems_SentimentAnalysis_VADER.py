"""
Text Problems - Sentiment Analysis - VADER

References:

"""

# Imports
from Utils.Utils import *

from vaderSentiment import vaderSentiment

# Init
VADER_SENTIMENT = vaderSentiment.SentimentIntensityAnalyzer()

# Main Functions
def Library_SentimentAnalysis_VADER(text, **params):
    '''
    Library - Sentiment Analysis - VADER
    '''
    # Run
    output = VADER_SENTIMENT.polarity_scores(text)
    OUT = {
        "negative": output["neg"],
        "neutral": output["neu"],
        "positive": output["pos"],
        "overall": output["compound"]
    }

    return OUT

# Main Vars
TASK_FUNCS = {
    "VADER": {
        "func": Library_SentimentAnalysis_VADER,
        "params": {}
    }
}