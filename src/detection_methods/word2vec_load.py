from gensim.models import KeyedVectors
from gensim.models.doc2vec import Doc2Vec



def init():
    global w2v_model
    global d2v_model
    #model = KeyedVectors.load_word2vec_format('/home/basha/Desktop/GoogleNews-vectors-negative300.bin', binary=True)

    w2v_model = KeyedVectors.load('C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\Word2VecModel\\word2vec.model')
    d2v_model = Doc2Vec.load('C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\Doc2VecModel\\doc2vec.model')
    #model = KeyedVectors.load_word2vec_format('/home/basha/Desktop/GoogleNews-vectors-negative300.bin', binary=True)
    #  w2v_model = KeyedVectors.load('/home/basha/PycharmProjects/DSA_Project/AutomaticFeartureEngineering/Datasets/Word2VecModel/word2vec.model')
   #    d2v_model = Doc2Vec.load('/home/basha/PycharmProjects/DSA_Project/AutomaticFeartureEngineering/Datasets/Doc2VecModel/enwiki_dbow/doc2vec.bin')
