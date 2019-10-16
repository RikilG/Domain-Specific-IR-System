from nltk.tokenize import wordpunct_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import os
import codecs
import threading, numpy

# TODO: remove dependancy on nltk and use stemmer:porter file and tokenizer:regex for wordpunct_tokenize

class Document:

    document_count = 0
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    stop_words.update( ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', 
                        '[', ']', '{', '}', '`', '``', "'s", "''", "m", "re", "s"
                        'es'])

    def __init__(self, filepath=None, doc_id=None, raw_data="", use_regex=False, stemming=True):
        """
        Attributes of Document object
            filepath: filepath to document
            raw_data: data present in document without any modifications
            data: normalized data
            words: list of all words present in document
            word_freq: dictionary of words and their frequency
        """
        self.filepath = filepath
        self.raw_data = raw_data
        if doc_id is None:
            self.doc_id = Document.document_count
        else:
            self.doc_id = doc_id
        Document.document_count += 1
        
        # Read data from file if object is not given any data
        if filepath is not None:
            with codecs.open(filepath, 'r', encoding="utf8",errors='ignore') as file:
                self.raw_data = file.read()
        
        # reduce caps and TODO:remove accents
        self.data = self.raw_data.lower()
        
        # tokenize words
        if use_regex is True:
            self.words = wordpunct_tokenize(self.data)
        else:
            self.words = word_tokenize(self.data)
        
        # remove stop words
        self.words = [word for word in self.words if word not in Document.stop_words]

        # run porter stemmer
        if stemming is True:
            self.words = [Document.stemmer.stem(word) for word in self.words]

        # create a dictionary to store words and their frequencies
        self.word_freq = dict()
        for word in self.words:
            self.word_freq[word] = (self.word_freq[word]+1) if word in self.word_freq else 1

        del self.data
        del self.words

    def __str__(self):
        return self.raw_data

def concurrent_read(files, document_list):
    for file in files:
        document_list.append(Document(file))
        print("Read: "+file)

# use this function to read complete corpus
def read_corpus(folderpath, threads=6):
    # TODO: check if folder path exists
    files = []
    document_list = []
    # r=root, d=directories, f=files
    for r, d, f in os.walk(folderpath):
        for filename in f:
            if '.txt' in filename:
                files.append(os.path.join(r,filename))
    
    # multithreaded IO
    # stats: 1-thread: 111sec, 4-threads: 70, 6-threads: 65sec 8-threads: 80sec
    # TODO: use multiprocessing to speed up (2 processes with 4 threads?!)
    #           be careful in setting doc_id in multiprocessing(give index in files list)
    # logger.log("using "+str(threads)+" threads", 6)
    # thread_count = threads # funciton parameter
    # thread_pool = list()
    # thread_files = numpy.array_split(numpy.array(files), thread_count)
    # concurrent_docs = [ [] for i in range(thread_count) ]
    # for i in range(thread_count):
    #     start = i + int(len(files)/thread_count)
    #     x = threading.Thread(target=concurrent_read, args=(thread_files[i], concurrent_docs[i]))
    #     thread_pool.append(x)
    #     x.start()
    
    # for index, thread in enumerate(thread_pool):
    #     thread.join()

    # for i in range(thread_count):
    #     for doc in concurrent_docs[i]:
    #         document_list.append(doc)
    
    # single threaded
    for file in files:
        document_list.append(Document(file))

    return document_list

def calc_collection_frequency(corpus):
    collection_freq = dict()
    for document in corpus:
        for word in document.word_freq:
            if word in collection_freq:
                collection_freq[word] += document.word_freq[word]
            else:
                collection_freq[word] = document.word_freq[word]
    return collection_freq

if __name__ == "__main__":
    print("***Testing Document class***")
    data = input("give data input: ")
    document = Document(raw_data=data)