import pandas as pd # Very NPC data frame library. Lets keep it simple!
import sklearn as sk # Very NPC ML library
import numpy as np # Yall better know this one!
import plotly.express as px # Good for making interactive plots
import nltk # Referenced in the assignment
from typing import Dict

nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from string import punctuation

def hello() -> str:
    return "Hello from fake-news!"

def load_dataset(path: str, n_rows: int) -> pd.DataFrame:
    print("Reading file...")
    df = pd.read_csv(path, low_memory=False, nrows=n_rows)

    # this is just a list of words we dont care about, so we can filter them out
    stop_words = set(stopwords.words('english')) | set(punctuation) | set("-'\"`’“”–—‘") | set(["''", "``"])

    print("Tokenizing titles...")
    # This tokenizes the text (evt google/ask chat about stemming and tokenization )
    stemmer = PorterStemmer()
    df["tokens"] = [[stemmer.stem(word) for word in word_tokenize(str(sent)) if stemmer.stem(word) not in stop_words] for sent in df["title"]]

    return df

def word_freq(df: pd.DataFrame, top_k: int) -> Dict[str, int]:
    # Count the top 20 most frequent words grouped by "type" (aka the training set label / prediction target)
    word_freq = {}
    for sent, label in zip(df["tokens"], df["type"]):
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
