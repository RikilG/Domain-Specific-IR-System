from irstructures.document import Document, read_corpus, calc_collection_frequency
from irstructures.invertedindex import InvertedIndex
import irstructures.models.boolean_retrieval as boolean_retrieval
import irstructures.models.vector_space as vector_space
import time
import os
from pandas import read_pickle
import pickle

def start_search(vsmodel, corpus, df, index):
    while True:
        query = input("Enter query: ")
        if query == "EXIT":
            break
        else:
            print("\nBoolean Retrieval results: ")
            output = boolean_retrieval.parse_query(query, corpus, index)
            for file in output:
                print(file)

            print("\nTf-Idf results: ")
            output = vector_space.parse_query(query, corpus, vsmodel, df)
            for file, prob in output:
                print(file, "\t", prob)
            
            print()

if __name__=='__main__':

    print("\n***Program started***\n")

    if("df.pickle" in os.listdir("./corpus")) and ("corpus.pickle" in os.listdir("./corpus")) and ("inv_index.pickle" in os.listdir("./corpus")):
        # folder name is corpus in this case

        # loading corpus
        start = time.time()
        with open("./corpus/corpus.pickle", "rb") as corpus_file:
            corpus = pickle.load(corpus_file)
        end = time.time()
        print("corpus loaded in: "+str(end - start))        
        
        # loading inverted index object
        start = time.time()
        with open("./corpus/inv_index.pickle", "rb") as inv_index_file:
            index = pickle.load(inv_index_file)
        end = time.time()
        print("inverted index loaded in: "+str(end - start))        

        # loading vectorspace object
        vsmodel = vector_space.Tf_Idf(inv_index=index)
        start = time.time()
        df = read_pickle('./corpus/df.pickle')
        end = time.time()
        print("dataframe pickle loaded in: "+str(end - start))        

    
    else:
        print("reading files")
        start = time.time()
        corpus = read_corpus('corpus')
        end = time.time()
        print("corpus generated in: "+str(end - start))

        # saving corpus object
        with  open("./corpus/corpus.pickle", "wb") as corpus_file:
            pickle.dump(corpus, corpus_file)

        collection_freq = calc_collection_frequency(corpus)

        print("Building inverted index")
        start = time.time()
        index = InvertedIndex(corpus)
        end = time.time()
        print("inverted index generated in: "+str(end - start))
        # saving inverted index object
        with open("./corpus/inv_index.pickle", "wb") as inv_index_file:
            pickle.dump(index, inv_index_file)

        print("Building vector space model")
        start = time.time()
        vsmodel = vector_space.Tf_Idf(inv_index=index)
        df = vsmodel.get_dataframe(corpus, collection_freq)
        df.to_pickle('./corpus/df.pickle')
        end = time.time()

        # saving vector space object

        print("vector space model built in: "+str(end - start))

    start_search(vsmodel, corpus, df, index)
    
    print("\n***End of program***\n")