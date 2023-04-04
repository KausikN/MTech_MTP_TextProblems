"""
Text Problems - Translation - Goslate

References:

"""

# Imports
from Utils.Utils import *

import goslate

# Init
GOSLATE_TRANSLATOR = goslate.Goslate()

# Main Functions
def Library_Translation_Goslate(text, target="de", **params):
    '''
    Library - Translation - Goslate
    '''
    # Run
    output = GOSLATE_TRANSLATOR.translate(text, target)
    OUT = {
        "text_target": output
    }

    return OUT

# Main Vars
TASK_FUNCS = {
    "Goslate": {
        "func": Library_Translation_Goslate,
        "params": {
            "target": "de"
        }
    }
}