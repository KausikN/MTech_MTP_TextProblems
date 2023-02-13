"""
Dataset Utils for Default Dataset

Link: None

Expected Files in Dataset Folder:
    - None
"""

# Imports


# Main Vars
DATASET_PATH = ""
DATASET_ITEMPATHS = {}
DATASET_DATA = {
    "Sentiment Analysis": {
        "output_type": {
            "type": "category",
            "categories": ["negative", "positive"]
        },
        "cols": {
            "all": ["text", "sentiment"],
            "keep": ["text"],
            "keep_default": ["text"],
            "target": "sentiment"
        }
    }
}
DATASET_PARAMS = {}
DATASET_SESSION_DATA = {}

# Main Functions

# Main Vars
DATASET_FUNCS = {
    "full": None,
    "train": None,
    "val": None,
    "test": None,

    "encode": None,
    "display": None
}