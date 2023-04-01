"""
Text Problems - Summarisation - Sumy

References:

"""

# Imports
from Utils.Utils import *
import functools

from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

# Init
SUMY_SUMMARIZERS = {
    "lex": LexRankSummarizer(),
    "luhn": LuhnSummarizer(),
    "lsa": LsaSummarizer(),
    "text_rank": TextRankSummarizer()
}

# Main Functions
def Library_Summarisation_Sumy(text, summarizer="lex", sentences_count=2, **params):
    '''
    Library - Summarisation - Sumy
    '''
    # Run
    text = PlaintextParser.from_string(text, Tokenizer("english")).document
    output = SUMY_SUMMARIZERS[summarizer](text, sentences_count=sentences_count)
    output = " ".join([str(line) for line in output])
    OUT = {
        "summary": output
    }

    return OUT

# Main Vars
TASK_FUNCS = {
    "Sumy - Lex": {
        "func": functools.partial(Library_Summarisation_Sumy, summarizer="lex"),
        "params": {
            "sentences_count": 2
        }
    },
    "Sumy - Luhn": {
        "func": functools.partial(Library_Summarisation_Sumy, summarizer="luhn"),
        "params": {
            "sentences_count": 2
        }
    },
    "Sumy - Lsa": {
        "func": functools.partial(Library_Summarisation_Sumy, summarizer="lsa"),
        "params": {
            "sentences_count": 2
        }
    },
    "Sumy - Text Rank": {
        "func": functools.partial(Library_Summarisation_Sumy, summarizer="text_rank"),
        "params": {
            "sentences_count": 2
        }
    }
}