from irstructures.document import Document, read_corpus, calc_collection_frequency
from irstructures.invertedindex import InvertedIndex
import irstructures.models.boolean_retrieval as boolean_retrieval
import irstructures.models.vector_space as vector_space
import os, time, pickle
from pandas import read_pickle

def start_search(vsmodel, corpus, df, index):
    while True:
        query = input("Enter query: ")
        if query == "EXIT":
            break
        else:
            print("\nBoolean Retrieval results: ")
            start = time.time()
            output = boolean_retrieval.parse_query(query, corpus, index)
            for fileid in output:
                print(corpus[fileid].filepath)
            end = time.time()
            print("returned in ", end-start, 's')

            print("\nTf-Idf results: ")
            start = time.time()
            output = vector_space.parse_query(query, corpus, vsmodel, df, output)
            for file, prob in output:
                print(file, "\t", prob)
            end = time.time()
            print("returned in ", end-start, 's')
            
            print()

if __name__=='__main__':

    print("\n***Program started***\n")

    if("df.pickle" in os.listdir("./pickle_files")) and ("corpus.pickle" in os.listdir("./pickle_files")) and ("inv_index.pickle" in os.listdir("./pickle_files")):
        # folder name is corpus in this case

        # loading corpus
        start = time.time()
        with open("./pickle_files/corpus.pickle", "rb") as corpus_file:
            corpus = pickle.load(corpus_file)
        end = time.time()
        print("corpus loaded in: "+str(end - start))        
        
        # loading inverted index object
        start = time.time()
        with open("./pickle_files/inv_index.pickle", "rb") as inv_index_file:
            index = pickle.load(inv_index_file)
        end = time.time()
        print("inverted index loaded in: "+str(end - start))        

        # loading vectorspace object
        vsmodel = vector_space.Tf_Idf(inv_index=index)
        start = time.time()
        df = read_pickle('./pickle_files/df.pickle')
        end = time.time()
        print("dataframe pickle loaded in: "+str(end - start))        

    
    else:
        print("reading files")
        start = time.time()
        corpus = read_corpus('corpus')
        end = time.time()
        print("corpus generated in: "+str(end - start))

        # saving corpus object
        with  open("./pickle_files/corpus.pickle", "wb") as corpus_file:
            pickle.dump(corpus, corpus_file)

        collection_freq = calc_collection_frequency(corpus)

        print("Building inverted index")
        start = time.time()
        index = InvertedIndex(corpus)
        end = time.time()
        print("inverted index generated in: "+str(end - start))
        # saving inverted index object
        with open("./pickle_files/inv_index.pickle", "wb") as inv_index_file:
            pickle.dump(index, inv_index_file)

        print("Building vector space model")
        start = time.time()
        vsmodel = vector_space.Tf_Idf(inv_index=index)
        # saving tf idf dataframe object
        df = vsmodel.get_dataframe(corpus, collection_freq)
        df.to_pickle('./pickle_files/df.pickle')
        end = time.time()


        print("vector space model built in: "+str(end - start))

    print('Size of dataframe: ', df.shape[0], 'tokens X', df.shape[1], 'docs')
    start_search(vsmodel, corpus, df, index)
    
    print("\n***End of program***\n")