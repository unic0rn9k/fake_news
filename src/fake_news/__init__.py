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


def tokenize_single_sentence(curr_sentence: str) -> List[str]:
    if not isinstance(curr_sentence, str):
        curr_sentence = str(curr_sentence) if curr_sentence is not None else ""
    
    tokens_in_sentences_not_stop = []
    for word in word_tokenize(curr_sentence):
        current_word = stemmer.stem(word.lower())  # Stem the word
        if current_word not in stop_words:  # Remove stopwords
            tokens_in_sentences_not_stop.append(current_word)
    return tokens_in_sentences_not_stop

def tokenize_old(sentences: List[str], print_reduction=True) -> List[List[str]]:
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

    if print_reduction:
        print(f"Redcuction rate: {reduction_rate * 100}%")

def tokenize(sentences: List[str]) -> List[List[str]]:
    with Pool() as pool:
        result = pool.map(tokenize_single_sentence, sentences)
    return result



def load_dataset(path: str, n_rows: int) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, nrows=n_rows)    


def word_freq_type(df: pd.DataFrame, top_k: int, col: str = "tokens") -> Dict[str, int]:
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

def word_freq(df: pd.DataFrame, top_k: int, col: str = "tokenized_content") -> Dict[str, int]:
    # Initialize a dictionary to store word frequencies
    word_freq = {}

    # Iterate through each row in the DataFrame
    for sent in df[col]:
        for word in sent:
            word_freq[word] = word_freq.get(word, 0) + 1

    # Sort the words by frequency and keep the top_k most frequent words
    word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:top_k])

    return word_freq



def count_urls(text):
    if isinstance(text, str):  # Ensure text is a valid string
        url_pattern = re.compile(r"https?://\S+|www\.\S+")  # Match http, https, www
        return len(url_pattern.findall(text))  # Count occurrences
    return 0  # Return 0 if text is NaN or not a string

def count_dates(text):
    if isinstance(text, str):  # Ensure text is a valid string
        date_pattern = re.compile(r"\b(?:\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4}|\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2})\b")  # Match http, https, www
        return len(date_pattern.findall(text))  # Count occurrences
    return 0  # Return 0 if text is NaN or not a string

def count_exclamations(text):
    return text.count("!") if isinstance(text, str) else 0

def count_uppercase_words(text):
    return sum(1 for word in text.split() if word.isupper())

def count_quotes(text):
    return text.count('"') + text.count("'")

def count_numbers(text):
    return len(re.findall(r"\b\d+\b", text))
    