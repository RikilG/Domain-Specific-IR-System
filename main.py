from irstructures.logger import Logger
from irstructures.document import Document, read_corpus, calc_collection_frequency
from irstructures.invertedindex import InvertedIndex
import irstructures.models.boolean_retrieval as boolean_retrieval
import irstructures.models.vector_space as vector_space
import time

print("\n***Program started***\n")

# using logger class to print statements based on level.
logger = Logger(6)
# folder name is corpus in this case
start = time.time()
corpus = read_corpus('corpus', logger.level, threads=4)
end = time.time()
logger.log("Total documents read: "+str(Document.document_count)+" in "+str(end-start))

logger.log("Calculating collection frequency", 1)
collection_freq = calc_collection_frequency(corpus)

logger.log("Writing word collection to disk", 1)
with open("word_collection.txt", 'w') as file:
    for word in collection_freq:
        file.write(word + "\n")

logger.log("Building inverted index", 1)
index = InvertedIndex(corpus)

logger.log("Building vector space model", 1)
start = time.time()
vsmodel = vector_space.Tf_Idf(corpus, collection_freq, inv_index=index)
end = time.time()
logger.log("vector space model built in: "+str(end - start))

# taking query input from user
while True:
    query = input("Enter query: ")
    if query == "EXIT":
        break;
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

print("\n***End of program***\n")