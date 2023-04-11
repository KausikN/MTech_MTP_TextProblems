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
# Dialogue
# from Methods.Dialogue import TextProblems_Dialogue_ChatterBot
# Summarisation
# from Methods.Summarisation import TextProblems_Summarisation_Gensim
from Methods.Summarisation import TextProblems_Summarisation_Sumy
# Translation
# from Methods.Translation import TextProblems_Translation_Goslate
from Methods.Translation import TextProblems_Translation_GoogleTrans
from Methods.Translation import TextProblems_Translation_TextBlob
# Question Answering
from Methods.QuestionAnswering import TextProblems_QuestionAnswering_Transformers

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
    "POS Tagging": {
    
    },
    "Relationship Extraction": {
        **TextProblems_RE_SpaCy.TASK_FUNCS
    },
    "Dialogue": {
        # **TextProblems_Dialogue_ChatterBot.TASK_FUNCS
    },
    "Summarisation": {
        # **TextProblems_Summarisation_Gensim.TASK_FUNCS,
        **TextProblems_Summarisation_Sumy.TASK_FUNCS
    },
    "Translation": {
        # **TextProblems_Translation_Goslate.TASK_FUNCS,
        **TextProblems_Translation_GoogleTrans.TASK_FUNCS,
        **TextProblems_Translation_TextBlob.TASK_FUNCS
    },
    "Question Answering": {
        **TextProblems_QuestionAnswering_Transformers.TASK_FUNCS
    }
}
TASK_OBJECTS = {
    "Dialogue": {
        # "ChatterBot": {
        #     "module": TextProblems_Dialogue_ChatterBot,
        #     "objects": TextProblems_Dialogue_ChatterBot.TASK_OBJECTS
        # }
    }
}

EXAMPLE_TEXTS = {
    "Sentiment Analysis": "The movie was average, but still better than the prequel.",
    "Named Entity Recognition": "Alan went to Paris to visit the palace with his friend Jerry.",
    "POS Tagging": "Alan went to Paris to visit the palace with his friend Jerry.",
    "Relationship Extraction": "Alan went to Paris to visit the palace with his friend Jerry.",
    "Dialogue": "Hello, how are you doing today?",
    "Summarisation": """Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to intelligence of humans and other animals. Example tasks in which this is done include speech recognition, computer vision, translation between (natural) languages, as well as other mappings of inputs. AI applications include advanced web search engines (e.g., Google Search), recommendation systems (used by YouTube, Amazon, and Netflix), understanding human speech (such as Siri and Alexa), self-driving cars (e.g., Waymo), generative or creative tools (ChatGPT and AI art), automated decision-making, and competing at the highest level in strategic game systems (such as chess and Go).[1] As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. For instance, optical character recognition is frequently excluded from things considered to be AI,[2] having become a routine technology.[3] Artificial intelligence was founded as an academic discipline in 1956, and in the years since it has experienced several waves of optimism,[4][5] followed by disappointment and the loss of funding (known as an "AI winter"),[6][7] followed by new approaches, success, and renewed funding.[5][8] AI research has tried and discarded many different approaches, including simulating the brain, modeling human problem solving, formal logic, large databases of knowledge, and imitating animal behavior. In the first decades of the 21st century, highly mathematical and statistical machine learning has dominated the field, and this technique has proved highly successful, helping to solve many challenging problems throughout industry and academia.[8][9] The various sub-fields of AI research are centered around particular goals and the use of particular tools. The traditional goals of AI research include reasoning, knowledge representation, planning, learning, natural language processing, perception, and the ability to move and manipulate objects.[a] General intelligence (the ability to solve an arbitrary problem) is among the field's long-term goals.[10] To solve these problems, AI researchers have adapted and integrated a wide range of problem-solving techniques, including search and mathematical optimization, formal logic, artificial neural networks, and methods based on statistics, probability, and economics. AI also draws upon computer science, psychology, linguistics, philosophy, and many other fields.""",
    "Translation": "Hello, how are you doing today?",
    "Question Answering": {
        "context": "My name is Alan. I am a student at the University of Toronto. I am studying computer science.",
        "question": "What is my name?"
    }
}