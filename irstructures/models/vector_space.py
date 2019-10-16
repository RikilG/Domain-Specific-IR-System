import numpy as np
from pandas import DataFrame
from ..document import Document

class Tf_Idf(DataFrame):
    def __init__(self, corpus, collection_freq, inverted_index=list()):
        #default constructor
        # TODO: name columns using doc_id instead of document indexes: useful if we are using multi-threading/processing
        #       i.e., columns = [ d.doc_id for d in corpus ]
        DataFrame.__init__(self,index=list(collection_freq.keys()), columns=[ d.doc_id for d in corpus ])

        self.inv_index=inverted_index
        for word in self.index:
            for i in range(len(corpus)):
                self.loc[word][i] = self.tf_idf(word, corpus[i], corpus)

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
        return np.dot(a,b)/np.sqrt(np.sum(a**2) * np.sum(b**2))

    def search(self, qdoc, corpus):
        """
            Input: Query(also a document)
            Returns: Sorted rank of documents according to the tf-idf value (in descending order) 
        """
        # TODO: To reduce searching time, do cos similarity with results of boolean retrieval
        #       since any way, we have to just order the results of boolean retrieval(we dont need to find 
        #       cos similarity with all docs/columns in dataframe)
        q_vec = np.ndarray((self.shape[0], ))
        for i,word in enumerate(self.index):
            q_vec[i] = self.tf_idf(word, qdoc, corpus)

        res = []
        for i in self.columns:
            temp = self.cosine_sim(q_vec, self[i])
            if temp>0:
                res.append((temp,i))
        
        return sorted(res, key=lambda x: x[0], reverse=True)
    

def parse_query(query, corpus, vsmodel):
    """
        Input: query, corpus(list of Document objects), vector space model
        Returns: list of relavent documents ranked w.r.t their score
    """
    # TODO: normalize vectors(unit vectors) to get score b/w 0 and 1, then we can use show that 
    #   as probability of match: OUT OF THE BOX CONCEPT 
    q = Document(raw_data=query)
    res = vsmodel.search(q, corpus)
    output = [ (corpus[i].filepath, score) for score, i in res ]
    return output