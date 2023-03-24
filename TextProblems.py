"""
Text Problems
"""

# Imports
from Utils.Utils import *
## Sentiment Analysis
from Methods.SentimentAnalysis import TextProblems_SentimentAnalysis_Pattern
from Methods.SentimentAnalysis import TextProblems_SentimentAnalysis_VADER
from Methods.SentimentAnalysis import TextProblems_SentimentAnalysis_TextBlob
## Named Entity Recognition
from Methods.NamedEntityRecognition import TextProblems_NER_SpaCy
from Methods.NamedEntityRecognition import TextProblems_NER_NLTK
# Relationship Extraction
from Methods.RelationshipExtraction import TextProblems_RE_SpaCy

# Main Functions

# Main Vars
TASK_MODULES = {
    "Sentiment Analysis": {
        **TextProblems_SentimentAnalysis_Pattern.TASK_FUNCS,
        **TextProblems_SentimentAnalysis_VADER.TASK_FUNCS,
        **TextProblems_SentimentAnalysis_TextBlob.TASK_FUNCS
    },
    "Named Entity Recognition": {
        **TextProblems_NER_SpaCy.TASK_FUNCS,
        **TextProblems_NER_NLTK.TASK_FUNCS
    },
    "Relationship Extraction": {
        **TextProblems_RE_SpaCy.TASK_FUNCS
    }
}