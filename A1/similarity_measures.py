from abc import abstractmethod
from collections import defaultdict
from math import log, sqrt


class CosineSimilarity:
    """
    This class calculates a similarity score between a given query and all documents in an inverted index.
    """
    def __init__(self, postings):
        self.postings = postings
        self.doc_to_norm = dict()
        self.set_document_norms()

    def __call__(self, query):
        doc_to_score = defaultdict(lambda: 0)
        self.get_scores(doc_to_score, query)
        return doc_to_score

    @abstractmethod
    def set_document_norms(self):
        """
        Set self.doc_to_norm to contain the norms of every document.
        """
        pass

    @abstractmethod
    def get_scores(self, doc_to_score, query):
        """
        For each document add an entry to doc_to_score with this document's similarity to query.
        """
        pass


class TF_Similarity(CosineSimilarity):
    def set_document_norms(self):
        for doc, token_counts in self.postings.doc_to_token_counts.items():
            self.doc_to_norm[doc] = sqrt(sum([tf ** 2 for tf in token_counts.values()]))

    def get_scores(self, doc_to_score, query):
        for token, query_term_frequency in query.items():
            for doc, document_term_frequency in self.postings.token_to_doc_counts[token].items():
                doc_to_score[doc] += query_term_frequency * document_term_frequency / self.doc_to_norm[doc]


class TFIDF_Similarity(CosineSimilarity):
    # TODO implement the set_document_norms and get_scores methods.
    # Get rid of the NotImplementedErrors when you are done.

    def set_document_norms(self):
        docNum = len(self.postings.doc_to_token_counts.keys())
        # calculate idf for each token
        token_idf = defaultdict(lambda: float(0))
        for token, document_term_default_dict in self.postings.token_to_doc_counts.items():
            df = len(document_term_default_dict.keys())
            if df != 0:
                token_idf[token] = log(docNum/df,2)

        for doc, token_counts in self.postings.doc_to_token_counts.items():
            doc_norm_list = []
            for token, tf in token_counts.items():
                doc_norm_list.append((tf*token_idf[token]) ** 2)
            self.doc_to_norm[doc] = sqrt(sum(doc_norm_list))



    def get_scores(self, doc_to_score, query):
        docNum = len(self.postings.doc_to_token_counts.keys())
        # calculate idf for each token
        token_idf = defaultdict(lambda: float(0))
        for token, document_term_default_dict in self.postings.token_to_doc_counts.items():
            df = len(document_term_default_dict.keys())
            if df != 0:
                token_idf[token] = log(docNum / df, 2)


        for token, query_term_frequency in query.items():
            for doc, document_term_frequency in self.postings.token_to_doc_counts[token].items():
                if self.doc_to_norm[doc] != 0:
                    doc_to_score[doc] += query_term_frequency * document_term_frequency * (token_idf[token] ** 2)/self.doc_to_norm[doc]
        # for eky in doc_to_score.keys():
        #     print(doc_to_score[eky])
