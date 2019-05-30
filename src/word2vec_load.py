from gensim.models import KeyedVectors


def init():
    global model
    #model = KeyedVectors.load_word2vec_format('/home/basha/Desktop/GoogleNews-vectors-negative300.bin', binary=True)
    model = KeyedVectors.load('/home/basha/PycharmProjects/DSA_Project/AutomaticFeartureEngineering/Datasets/Word2VecModel/word2vec.model')
