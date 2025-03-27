import pandas as pd # Very NPC data frame library. Lets keep it simple!
import sklearn as sk # Very NPC ML library
import numpy as np # Yall better know this one!
import plotly.express as px # Good for making interactive plots
import nltk # Referenced in the assignment
from typing import Dict, List
from multiprocessing import Pool


import re
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from typing import Dict, List
from multiprocessing import Pool

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from string import punctuation

# Define the stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english')) | set(punctuation) | set("-'\"`’“”–—‘") | set(["''", "``"])


nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from string import punctuation

def load_dataset(path: str, n_rows: int) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, nrows=n_rows)

def tokenize_single_sentence(curr_sentence: str) -> List[str]:
    if not isinstance(curr_sentence, str):
        curr_sentence = str(curr_sentence) if curr_sentence is not None else ""
    
    tokens_in_sentences_not_stop = []
    for word in word_tokenize(curr_sentence):
        current_word = stemmer.stem(word.lower())  # Stem the word
        if current_word not in stop_words:  # Remove stopwords
            tokens_in_sentences_not_stop.append(current_word)
    return tokens_in_sentences_not_stop

def tokenize(sentences: List[str], print_reduction=True) -> List[List[str]]:
    with Pool() as pool:
        result = pool.map(tokenize_single_sentence, sentences)
    return result

def word_freq(df: pd.DataFrame, top_k: int, col: str = "tokens") -> Dict[str, int]:
    # Count the top 20 most frequent words grouped by "type" (aka the training set label / prediction target)
    word_freq = {}
    for sent, label in zip(df[col], df["type"]):
        if label == "NaN":
            continue
        for word in sent:
            word_freq[label] = word_freq.get(label, {})
            word_freq[label][word] = word_freq[label].get(word, 0) + 1

    word_freq = {
        key: dict(sorted(sub.items(), key=lambda item: item[1], reverse=True)[:top_k])
        for key, sub in word_freq.items()
    }

    return word_freq

