"""
Text Problems - Relationship Extraction - SpaCy

References:

"""

# Imports
from Utils.Utils import *

# Init
SPACY_RELATION_PATTERNS = [
    {"DEP": "ROOT"}, 
    {"DEP": "prep", "OP": "?"},
    {"DEP": "agent", "OP": "?"},  
    {"POS": "ADJ", "OP": "?"}
]

# Util Functions
def find_root_of_sentence(doc):
    root_token = None
    for token in doc:
        if (token.dep_ == "ROOT"):
            root_token = token
    return root_token

def contains_root(verb_phrase, root):
    vp_start = verb_phrase.start
    vp_end = verb_phrase.end
    if (root.i >= vp_start and root.i <= vp_end):
        return True
    else:
        return False
    
def get_verb_phrases(doc):
    root = find_root_of_sentence(doc)
    # Find relation phrases
    # verb_phrases = textacy.extract.matches.token_matches(doc, SPACY_VERB_PATTERNS)
    matcher = spacy.matcher.Matcher(NLP.vocab)
    matcher.add("matching_1", [SPACY_RELATION_PATTERNS])
    matches = matcher(doc)
    span = doc[matches[-1][1]:matches[-1][2]]
    verb_phrases = [span]
    # Check if contains root
    new_vps = []
    for verb_phrase in verb_phrases:
        if (contains_root(verb_phrase, root)):
            new_vps.append(verb_phrase)
    return new_vps

def longer_verb_phrase(verb_phrases):
    longest_length = 0
    longest_verb_phrase = None
    for verb_phrase in verb_phrases:
        if len(verb_phrase) > longest_length:
            longest_verb_phrase = verb_phrase
    return longest_verb_phrase

def find_noun_phrase(verb_phrase, noun_phrases, side):
    for noun_phrase in noun_phrases:
        if (side == "left" and noun_phrase.start < verb_phrase.start):
            return noun_phrase
        elif (side == "right" and noun_phrase.start > verb_phrase.start):
            return noun_phrase

def find_triplet(sentence):
    doc = NLP(sentence)
    verb_phrases = get_verb_phrases(doc)
    noun_phrases = doc.noun_chunks
    # --- Only 1 verb phrase
    verb_phrase = None
    if len(verb_phrases) == 0:
        return ("", "", "")
    elif (len(verb_phrases) > 1):
        verb_phrase = longer_verb_phrase(list(verb_phrases))
    else:
        verb_phrase = verb_phrases[0]
    left_noun_phrase = find_noun_phrase(verb_phrase, noun_phrases, "left")
    right_noun_phrase = find_noun_phrase(verb_phrase, noun_phrases, "right")

    return (left_noun_phrase, verb_phrase, right_noun_phrase)

def find_triplets(sentence):
    doc = NLP(sentence)
    verb_phrases = get_verb_phrases(doc)
    noun_phrases = doc.noun_chunks
    # --- Multiple verb phrases
    triplets = []
    for verb_phrase in verb_phrases:
        left_noun_phrase = find_noun_phrase(verb_phrase, noun_phrases, "left")
        right_noun_phrase = find_noun_phrase(verb_phrase, noun_phrases, "right")
        triplets.append((left_noun_phrase, verb_phrase, right_noun_phrase))

    return triplets

# Main Functions
def Library_RE_SpaCy(text, **params):
    '''
    Library - Relationship Extraction - SpaCy
    '''
    # Init
    sentences = NLP(text).sents
    # Run
    output = []
    for sentence in sentences: output.extend(find_triplets(str(sentence))) 
    OUT = {
        "relations": [
            {
                "subject": str(output[i][0]),
                "relation": str(output[i][1]),
                "object": str(output[i][2])
            }
            for i in range(len(output))
        ]
    }

    return OUT

# Main Vars
TASK_FUNCS = {
    "SpaCy": {
        "func": Library_RE_SpaCy,
        "params": {}
    }
}