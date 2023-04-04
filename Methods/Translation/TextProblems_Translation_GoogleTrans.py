"""
Text Problems - Translation - Goslate

References:

"""

# Imports
from Utils.Utils import *

from googletrans import Translator as GoogleTrans_Translator

# Init
GOOGLETRANS_TRANSLATOR = GoogleTrans_Translator()

# Main Functions
def Library_Translation_GoogleTrans(text, target="de", **params):
    '''
    Library - Translation - Google Translate
    '''
    # Run
    output = GOOGLETRANS_TRANSLATOR.translate(text, dest=target)
    OUT = {
        "text_target": output.text
    }

    return OUT

# Main Vars
TASK_FUNCS = {
    "Goslate": {
        "func": Library_Translation_GoogleTrans,
        "params": {
            "target": "de"
        }
    }
}