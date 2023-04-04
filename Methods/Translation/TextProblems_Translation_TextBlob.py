"""
Text Problems - Translation - TextBlob

References:

"""

# Imports
from Utils.Utils import *

from textblob import TextBlob

# Main Functions
def Library_Translation_TextBlob(text, target="de", **params):
    '''
    Library - Translation - TextBlob
    '''
    # Run
    output = TextBlob(text)
    from_lang = str(output.detect_language())
    output = output.translate(from_lang=from_lang, to=target)
    OUT = {
        "text_target": str(output)
    }

    return OUT

# Main Vars
TASK_FUNCS = {
    "TextBlob": {
        "func": Library_Translation_TextBlob,
        "params": {
            "target": "de"
        }
    }
}