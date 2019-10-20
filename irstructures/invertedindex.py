from .document import Document

# TODO: add this class to boolean_retrieval.py as it is only used there.
class InvertedIndex(dict):
    """
    Class to generate and hold inverted index. This class extends the inbuilt dictionary data structure.

    This object when created, generates inverted index.
    """
    
    def __init__(self, corpus, collection_freq=None):
        """Creating posting lists in inverted index for each word
        
        Parameters
        ----------
        corpus: list
            list of document objects
        collection_freq: dict
            dictionary of each word and its corpus frequency
        """

        for document in corpus:
            for word in document.word_freq:
                if word in self:
                    self[word].append(document.doc_id)
                else:
                    self[word] = [document.doc_id]