import numpy as np
from pandas import DataFrame
from ..document import Document
import time

class Tf_Idf():
    def __init__(self, inv_index=list()):
        #default constructor
        # TODO: name columns using doc_id instead of document indexes: useful if we are using multi-threading/processing
        #       i.e., columns = [ d.doc_id for d in corpus ]
        self.inv_index=inv_index

    def get_dataframe(self, corpus, collection_freq):
        """
            This method computes fills tf_idf scores and returns 
            the dataframe.
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
        """
            Returns the frequency of the word as logarithm(No of occurences in the document)
            by using the word_freq dictionary  
        """
        if word in document.word_freq:
            return (1+np.log10(document.word_freq[word]))
        else:
            return 0
    
    def doc_freq(self, word, corpus):
        """
            Returns the count of all the documents(which are part of corpus) 
            in which the word occurs
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
        """
        Returns the Inverse Document Frequency (idf) of a word 
          idf = Logarithm ((Total Number of Documents) /  
            (Number of documents containing the word)) 
        """
        idf = self.doc_freq(word, corpus)
        if idf == 0: 
            return 0
        return np.log10(len(corpus)/(idf))           
    
    def tf_idf(self, word, document, corpus):
        """
            Returns the calculated tf-idf score
            tf_idf(word, document) = term frequency(word, document)* inverse document freq(word, document)
        """
        return self.term_freq(word, document)*self.idf(word, corpus)
    
    def cosine_sim(self, a, b):
        """
            Returns the cosine or the dot product of two vectors(query and document or
            document and document)
        """
        denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
        if denom==0: return 0
        return np.dot(a,b)/denom

    def search(self, qdoc, corpus, df, boolean_output):
        """
            Input: Query(also a document)
            Returns: Sorted rank of documents according to the tf-idf value (in descending order) 
        """
        # TODO: To reduce searching time, do cos similarity with results of boolean retrieval
        #       since any way, we have to just order the results of boolean retrieval(we dont need to find 
        #       cos similarity with all docs/columns in dataframe)
        q_vec = np.ndarray((df.shape[0], ))
        for i,word in enumerate(df.index):
            q_vec[i] = self.tf_idf(word, qdoc, corpus)

        res = []
        for col in boolean_output:
            temp = self.cosine_sim(q_vec, df[col])
            if temp>0:
                res.append((temp,col))
        
        return sorted(res, key=lambda x: x[0], reverse=True)
    

def parse_query(query, corpus, vsmodel, df, boolean_output):
    """
        Input: query, corpus(list of Document objects), vector space model
        Returns: list of relavent documents ranked w.r.t their score
    """
    # TODO: normalize vectors(unit vectors) to get score b/w 0 and 1, then we can use show that 
    #   as probability of match: OUT OF THE BOX CONCEPT 
    q = Document(raw_data=query)
    res = vsmodel.search(q, corpus, df, boolean_output)
    output = [ (corpus[i].filepath, score) for score, i in res ]
    return output