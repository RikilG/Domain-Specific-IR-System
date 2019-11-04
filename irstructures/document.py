from nltk.tokenize import wordpunct_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import os
import codecs

# TODO: remove dependancy on nltk and use stemmer:porter file and tokenizer:regex for wordpunct_tokenize

class Document:
    """
    Class used to represent Document files

    Attributes
    ----------
    filepath: str
        path to file on disk (can be relative)
    doc_id: int
        unique document id for each document
    word_freq: dict
        dictionary with each word and it's corresponding frequency
    """

    document_count = 0
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    stop_words.update( ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', 
                        '[', ']', '{', '}', '`', '``', "'s", "''", "m", "re", "s"
                        'es'])
    

    def __init__(self, filepath=None, doc_id=None, raw_data="", use_regex=False, stemming=True):

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


# use this function to read complete corpus
def read_corpus(folderpath):
    """simple function to read documents from given folder

    Parameters
    ----------
    folderpath: str
        path to corpus/folder containing documents on disk
    
    Returns
    -------
    list
        list of Document objects each representing a document
    """

    if(not os.path.exists(folderpath)):
        raise Exception(f"Given folder path: '{folderpath}' does not exist")
        return

    files = []
    document_list = []
    # r=root, d=directories, f=files
    for r, d, f in os.walk(folderpath):
        for filename in f:
            # if len(files)>=25:break
            if '.txt' in filename:
                files.append(os.path.join(r,filename))
    
    for file in files:
        document_list.append(Document(file))
        print(f"file read: {file}")

    return document_list


def calc_collection_frequency(corpus):
    """calculates frequency each word in entire corpus

    Parameters
    ----------
    corpus: list
        list of Document objects
    
    Returns
    -------
    dict
        dictionary of each word and its corpus frequency
    """
    
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