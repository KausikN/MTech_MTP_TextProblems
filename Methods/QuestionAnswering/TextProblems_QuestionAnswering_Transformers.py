"""
Text Problems - Question Answering - Transformers

References:

"""

# Imports
from Utils.Utils import *

from transformers import pipeline

# Init
QA_MODEL = pipeline("question-answering")

# Main Functions
def Library_QuestionAnswering_Transformers(context, question, **params):
    '''
    Library - Question Answering - Transformers
    '''
    # Run
    output = QA_MODEL(context=context, question=question)
    OUT = {
        "answer": str(output["answer"]),
        "score": float(output["score"])
    }

    return OUT

# Main Vars
TASK_FUNCS = {
    "Transformers": {
        "func": Library_QuestionAnswering_Transformers,
        "params": {}
    }
}