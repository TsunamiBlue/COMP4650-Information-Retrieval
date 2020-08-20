import nltk
from functools import lru_cache


class Preprocessor:
    def __init__(self):
        # Stemming is the most time-consuming part of the indexing process, we attach a lru_cache to the stermmer
        # which will store upto 100000 stemmed forms and reuse them when possible instead of applying the
        # stemming algorithm.
        self.stem = lru_cache(maxsize=10000)(nltk.stem.PorterStemmer().stem)
        # self.tokenize = nltk.tokenize.WhitespaceTokenizer().tokenize
        self.tokenize = nltk.tokenize.TreebankWordTokenizer().tokenize

    def __call__(self, text):
        # tokens = nltk.tokenize.WhitespaceTokenizer().tokenize(text)
        tokens = nltk.tokenize.TreebankWordTokenizer().tokenize(text)
        tokens = [self.stem(token) for token in tokens]
        return tokens
