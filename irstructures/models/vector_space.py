import numpy as np
from pandas import DataFrame
from ..document import Document
import time

class Tf_Idf():
    """
    Class used to represent Tf-Idf model

    Attributes
    ----------
    inv_index: dict
        inverted index containing words as keys and list of documents os values
    
    Methods
    -------
    get_dataframe(self, corpus, collection_freq)
        This method computes tf_idf scores and returns the dataframe
    term_freq(self, word, document)
        Returns the logarithm of frequency of the word(No of occurences in the document)
    doc_freq(self, word, corpus)
        Returns the count of all the documents in which the word occurs
    idf(self, word, corpus)
        Returns the Inverse Document Frequency (idf) of a word
    tf_idf(self, word, document, corpus)
        Returns the calculated tf-idf score
    cosine_sim(self, a, b)
        Returns the cosine or the dot product of two vectors
    search(self, qdoc, corpus, df, boolean_output)
        find documents which match query and rank them
    """

    def __init__(self, inv_index=list()):
        #default constructor
        self.inv_index=inv_index

    def get_dataframe(self, corpus, collection_freq):
        """This method computes tf_idf scores and returns the dataframe

        Parameters
        ----------
        corpus: list
            list containing Document class objects
        collection_freq: dict
            dictionary with words as keys and their overall frequency in corpus as values

        Returns
        -------
        pandas.DataFrame
            dataframe containing tf-idf scores for each word in a document
        """

        start = time.time()
        df = DataFrame(index=list(collection_freq.keys()), columns=[ d.doc_id for d in corpus ])
        end = time.time()
        print("Data Frame initialized in ", end-start)
        start = time.time()
        for word in df.index:
            for i in range(len(corpus)):
                df.loc[word][i] = self.tf_idf(word, corpus[i], corpus)
        end = time.time()
        print("Data Frame made in ", end-start)
        return df

    def term_freq(self, word, document):
        """Returns the logarithm of frequency of the word(No of occurences in the document)

        calculates 1 + log10( word_freq[word] ). if word frequency is 0, then it returns 0

        Parameters
        ----------
        word: str
            word whose frequency of occurence must be calculated
        document: Document
            corresponding document in which word frequency is calculated
        
        Returns
        -------
        float
            logarithm of frequency of word
        """

        if word in document.word_freq:
            return (1+np.log10(document.word_freq[word]))
        else:
            return 0
    
    def doc_freq(self, word, corpus):
        """Returns the count of all the documents in which the word occurs

        Parameters
        ----------
        word: str
            word whose frequency of occurence must be calculated
        corpus: list
            list containing Document class objects
        
        Returns
        -------
        int
            count of all the documents in which the word occurs
        """

        # finding doc_freq using len(posting list) from inverted index(if present)
        if len(self.inv_index)>0:
            return len(self.inv_index[word])

        count = 0
        for doc in corpus:
            if word in doc.word_freq:
                count += 1
        return count

    def idf(self, word, corpus):
        """Returns the Inverse Document Frequency (idf) of a word 
        
        idf = Logarithm ((Total Number of Documents) /  (Number of documents containing the word))

        Parameters
        ----------
        word: str
            word whose frequency of occurence must be calculated
        corpus: list
            list containing Document class objects

        Returns
        -------
        float
            inverse log of calculated document frequency
        """

        idf = self.doc_freq(word, corpus)
        if idf == 0: 
            return 0
        return np.log10(len(corpus)/(idf))           
    
    def tf_idf(self, word, document, corpus):
        """Returns the calculated tf-idf score
        
        tf_idf(word, document) = term frequency(word, document)* inverse document freq(word, document)

        Parameters
        ----------
        word: str
            word whose frequency of occurence must be calculated
        document: Document
            corresponding document in which word frequency is calculated
        corpus: list
            list containing Document class objects
        """

        return self.term_freq(word, document)*self.idf(word, corpus)
    
    def cosine_sim(self, a, b):
        """Returns the cosine or the dot product of two vectors

        Parameters
        ----------
        a: numpy.array
            first document vector
        b: numpy.array
            second document vector
        
        Returns
        -------
        float
            cosine or the dot product of given vectors
        """

        denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
        if denom==0: return 0
        return np.dot(a,b)/denom

    def search(self, qdoc, corpus, vs_dataframe, boolean_output):
        """Find documents which match query and rank them

        Parameters
        ----------
        qdoc: Document
            Document object which is generated corresponding to input query
        corpus: list
            list containing Document class objects
        vs_dataframe: pandas.DataFrame
            dataframe containing all tf-idf scores for each word
        boolean_output: list
            list of file_id's as given by boolean retrieval model
            
        Returns
        -------
        list
            Ranked document list sorted according to the tf-idf value (descending order) 
        """

        q_vec = np.ndarray((vs_dataframe.shape[0], ))
        for i,word in enumerate(vs_dataframe.index):
            q_vec[i] = self.tf_idf(word, qdoc, corpus)

        res = []
        for col in boolean_output:
            temp = self.cosine_sim(q_vec, vs_dataframe[col])
            if temp>0:
                res.append((temp,col))
        
        return sorted(res, key=lambda x: x[0], reverse=True)
    

def parse_query(query, corpus, vsmodel, vs_dataframe, boolean_output):
    """This function parses the query and returns relavent files

    Parameters
    ----------
    query: str
        input query string
    corpus: list
        list containing Document class objects
    vsmodel: Tf_Idf
        object containing vector space model 
    vs_dataframe: pandas.Dataframe
        dataframe containing Tf-Idf values for each word
    boolean_output: list
        list of file_id's as given by boolean retrieval model

    Returns
    -------
    list
        relavent documents ranked w.r.t their score
    """

    q = Document(raw_data=query)
    res = vsmodel.search(q, corpus, vs_dataframe, boolean_output)
    output = [ (corpus[i].filepath, score) for score, i in res ]
    return output