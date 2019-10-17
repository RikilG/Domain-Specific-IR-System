from irstructures.document import Document, read_corpus, calc_collection_frequency
from irstructures.invertedindex import InvertedIndex
import irstructures.models.boolean_retrieval as boolean_retrieval
import irstructures.models.vector_space as vector_space
import time
import os
import pickle

def start_search(vsmodel, corpus):
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
            output = vector_space.parse_query(query, corpus, vsmodel)
            for file, prob in output:
                print(file, "\t", prob)
            
            print()

if __name__=='__main__':

    print("\n***Program started***\n")

    if("vectorspace.pickle" in os.listdir("./corpus")) and ("corpus.pickle" in os.listdir("./corpus")) and ("inv_index.pickle" in os.listdir("./corpus")):
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
        start = time.time()
        with open('./corpus/vectorspace.pickle', 'rb') as vector_space_file:
            vsmodel = pickle.load(vector_space_file)
        end = time.time()
        print("vectorspace loaded in: "+str(end - start))        

    
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
        vsmodel = vector_space.Tf_Idf(corpus, collection_freq, inv_index=index)
        end = time.time()

        # saving vector space object
        with open('./corpus/vectorspace.pickle', 'wb') as vector_space_file:
            pickle.dump(vsmodel, vector_space_file)

        print("vector space model built in: "+str(end - start))

    start_search(vsmodel, corpus)
    
    print("\n***End of program***\n")