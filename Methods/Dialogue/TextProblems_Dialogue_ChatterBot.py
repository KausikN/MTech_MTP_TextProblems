"""
Text Problems - Dialogue - ChatterBot

References:

"""

# Imports
from Utils.Utils import *

import chatterbot
from chatterbot.trainers import ChatterBotCorpusTrainer

# Main Functions
def Dialogue_Train_ChatterBot():
    global TASK_OBJECTS
    if TASK_OBJECTS["chatbot"] is None:
        TASK_OBJECTS["chatbot"] = chatterbot.ChatBot("Chatbot")
        CHATTERBOT_TRAINER = ChatterBotCorpusTrainer(TASK_OBJECTS["chatbot"])
        CHATTERBOT_TRAINER.train("chatterbot.corpus.english")

def Library_Dialogue_ChatterBot(text, **params):
    '''
    Library - Dialogue - ChatterBot
    '''
    # Run
    output = TASK_OBJECTS["chatbot"].get_response(text)
    OUT = {
        "reply": str(output)
    }

    return OUT

# Main Vars
TASK_FUNCS = {
    "ChatterBot": {
        "func": Library_Dialogue_ChatterBot,
        "params": {}
    }
}
TASK_OBJECTS_LOAD_FUNCS = {
    "chatbot": Dialogue_Train_ChatterBot
}
TASK_OBJECTS = {
    "chatbot": None
}