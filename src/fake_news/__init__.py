import pandas as pd # Very NPC data frame library. Lets keep it simple!
import sklearn as sk # Very NPC ML library
import numpy as np # Yall better know this one!
import plotly.express as px # Good for making interactive plots
import nltk # Referenced in the assignment
from typing import Dict, List

nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from string import punctuation

def tokenize(sentences: List[str]) -> List[List[str]]:
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english')) | set(punctuation) | set("-'\"`’“”–—‘") | set(["''", "``"])
    count_with_stop = 0
    count_without_stop = 0
    result = []
    for curr_sentence in sentences:
        tokens_in_sentences_not_stop = []
        for word in word_tokenize(str(curr_sentence)):
            current_word = stemmer.stem(word)
            if current_word not in stop_words:
                tokens_in_sentences_not_stop.append(current_word)
                count_without_stop += 1
            count_with_stop += 1
        result.append(tokens_in_sentences_not_stop)

    reduction_rate = (count_with_stop-count_without_stop)/count_with_stop
    print(f"Redcuction rate: {reduction_rate * 100}%")

    return result

def load_dataset(path: str, n_rows: int) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, nrows=n_rows)

#def toeknize(text_col: List[str]) -> List[List[str]]:
    


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

