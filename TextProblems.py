"""
Text Problems

Embedding/Encoding Stage
Inputs:
 - Input Text : Text
 - True Label / Target : Label / Target in any form (Text, Categorical, etc.)
Outputs:
 - Encoded Text : Encoded Text (One-Hot Vectors) (Text_Length, N_Unique_Words)
 - Encoded Label / Target : Encoded Label / Target (One-Hot Vectors)

Training Stage
Inputs:
 - Encoded Text : Encoded Text (One-Hot Vectors) (Text_Length, N_Unique_Words)
 - Encoded Label / Target : Encoded Label / Target (One-Hot Vectors)
Outputs:
 - Trained Model : Model that can be used to predict output from input text

Testing Stage
Inputs:
 - Trained Model : Model that can be used to predict output from input text
 - Encoded Text : Encoded Text (One-Hot Vectors) (Text_Length, N_Unique_Words)
Outputs:
 - Output : Output from the model (Text, Categorical, etc.)
"""

# Imports
# Segmenter Imports
from Methods.SentimentAnalysis import TextProblems_SentimentAnalysis_XLNet
from Methods.SentimentAnalysis import TextProblems_SentimentAnalysis_Bert
# Dataset Imports
from Data.Datasets.DefaultDataset import DatasetUtils as DatasetUtils_Default
from Data.Datasets.IMDB import DatasetUtils as DatasetUtils_IMDB

# Main Functions

# Main Vars
TASK_MODULES = {
    "Sentiment Analysis": {
        "XLNet": {
            **TextProblems_SentimentAnalysis_XLNet.TASK_FUNCS
        },
        "BERT": {
            **TextProblems_SentimentAnalysis_Bert.TASK_FUNCS
        },
    }
}

DATASETS = {
    "IMDB": DatasetUtils_IMDB
}
DATASET_DEFAULT = DatasetUtils_Default