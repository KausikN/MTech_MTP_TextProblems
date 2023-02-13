"""
Encode Utils
"""

# Imports
import os
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap, LocallyLinearEmbedding
# Tokenizers
from tensorflow.keras.preprocessing.text import Tokenizer
# Embeddings
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Main Functions
## Encode Functions
def EncodeUtils_EncodeText(data, tokenizer_data, embedding_data, DATASET_SESSION_DATA={}):
    '''
    EncodeUtils - Encode Text
    '''
    # Init
    data = np.array(data)
    ## Tokenize
    if DATASET_SESSION_DATA["tokenizer"] is None:
        TOKENIZER = Tokenizer(**tokenizer_data["params"])
        TOKENIZER.fit_on_texts(data)
        DATASET_SESSION_DATA["tokenizer"] = TOKENIZER
    data_tokenized = TOKENIZER.texts_to_sequences(data)
    ## Embed
    data_embedded = pad_sequences(data_tokenized, **embedding_data["params"])
    
    OutData = {
        "data": data_embedded,
        "tokenizer": TOKENIZER,
        "embedding": None
    }
    return OutData, DATASET_SESSION_DATA

## Norm Functions
def EncodeUtils_NormData_MinMax(data, min_val=0.0, max_val=1.0):
    '''
    EncodeUtils - Norm Data - MinMax
    '''
    # Init
    data = np.array(data)
    if min_val == max_val: return data
    # Norm
    return (data - min_val) / (max_val - min_val)

# Main Vars
DR_METHODS = {
    None: None,
    "PCA": PCA,
    "SVD": TruncatedSVD,
    "LDA": LinearDiscriminantAnalysis,
    "ISOMAP": Isomap,
    "LLE": LocallyLinearEmbedding
}