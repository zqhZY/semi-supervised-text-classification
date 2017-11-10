# -*- coding:utf-8 -*-
import argparse
import codecs
import cPickle
import os
import keras
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint
from gensim.models.word2vec import Word2Vec
import numpy as np
import sys
from self_train import self_train
from models.model import get_model
#from self_train import self_train

reload(sys)
sys.setdefaultencoding("utf-8")

parser = argparse.ArgumentParser(description='text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch_size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log_interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test_interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save_interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save_dir', type=str, default='../checkpoint', help='where to save the snapshot')

# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch' )
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max_norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-max_seq_len', type=int, default=600, help='number of embedding dimension [default: 128]')
parser.add_argument('-embed_dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel_num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel_sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')

# model rnn
parser.add_argument('-recurrent_dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-rnn_max_seq_len', type=int, default=600, help='number of embedding dimension [default: 128]')
parser.add_argument('-rnn_embed_dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-rnn_num_lstm', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-rnn_num_dense', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-rnn_static', action='store_true', default=False, help='fix the embedding')

# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no_cuda', action='store_true', default=False, help='disable the gpu' )
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
parser.add_argument('-mode', type=str, default="cnn", help='which mode to run')
parser.add_argument('-train_num', type=int, default=-1, help='which mode to run')
parser.add_argument('-threshold', type=float, default=0.8, help='which mode to run')
args = parser.parse_args()


DATA_DIR = '../data/news_data/'
#EMBEDDING_FILE = '../../word2vector/model/w2v_train_with_fish.mod'
EMBEDDING_FILE = '../../word2vector/model/news_data_w2v_min5_mod'
TRAIN_DATA_FILE = DATA_DIR + 'news_data_train_rm_low_freq_cleaned.csv'
TEST_DATA_FILE = DATA_DIR + 'news_data_test_rm_low_freq_cleaned.csv'
MAX_SEQUENCE_LENGTH = args.max_seq_len
EMBEDDING_DIM = 128
VALIDATION_SPLIT = 0.1
MAX_NB_WORDS = 300000



########################################
# process texts in datasets
########################################
print('Processing text dataset')

save = False
load_tokenizer = True
save_path = "../checkpoint"
tokenizer_name = "tokenizer.pkl"


train_set = []
int_labels = []
count = 0
#label_map = {u"办理":0, u"投诉（含抱怨）":1, u"咨询（含查询）":2, u"其他":3, u"表扬及建议":4}
label_map = {u"体育":0,  u"彩票":1,  u"时尚":2,  u"游戏":3,  u"股票":4, u"娱乐":5,  u"房产":6,  u"时政":7,  u"社会":8,  u"财经":9, u"家居":10, u"教育":11,  u"星座":12,  u"科技":13}
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    for line in f:
        # print  line
        count += 1
        tokens = line.strip().split(",")
        try:
            int_labels.append(label_map[tokens[2]])
            train_set.append(tokens[1].lower())
        except Exception, e:
            print "--------------", e.message

categorical_labels = to_categorical(int_labels, num_classes=14)
print type(categorical_labels)
print('Found %s texts in train' % len(train_set))


'''
this part is solve keras.preprocessing.text can not process unicode
start here
'''


def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    if lower: text = text.lower()
    if type(text) == unicode:
        translate_table = {ord(c): ord(t) for c, t in zip(filters, split * len(filters))}
    else:
        translate_table = keras.maketrans(filters, split * len(filters))
    text = text.translate(translate_table)
    seq = text.split(split)
    return [i for i in seq if i]


keras.preprocessing.text.text_to_word_sequence = text_to_word_sequence
'''
end here
'''

if load_tokenizer:
    print('Load tokenizer...')
    tokenizer = cPickle.load(open(os.path.join(save_path, tokenizer_name), 'rb'))
else:
    print("Fit tokenizer...")
    tokenizer = Tokenizer(lower=True)
    tokenizer.fit_on_texts(train_set)
    if save:
        print("Save tokenizer...")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cPickle.dump(tokenizer, open(os.path.join(save_path, tokenizer_name), "wb"))

sequences_train = tokenizer.texts_to_sequences(train_set)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

#train_data = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
train_data = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)[:40000] # for speed up

if args.train_num != -1:
    labled_num = args.train_num
    labeled_train_set = train_data[:labled_num]
    train_set_labels = categorical_labels[:labled_num]
    unlabel_set = train_data[labled_num:]

print('Shape of labeled data tensor:', labeled_train_set.shape)
print('Shape of label tensor:', categorical_labels.shape)
print('Shape of unlabeled data tensor:', unlabel_set.shape)
#X_train, X_dev, y_train, y_dev = train_test_split(train_data, categorical_labels, test_size=0.1, random_state=42)
#np.save("../checkpoint/train_data.npy", train_data)
#np.save("../checkpoint/train_labels.npy", categorical_labels)


########################################
# prepare embeddings
########################################
print('Preparing embedding matrix')
word2vec = Word2Vec.load(EMBEDDING_FILE)

nb_words = min(MAX_NB_WORDS, len(word_index)) + 1

embedding_matrix = np.zeros((nb_words, args.embed_dim))
for word, i in word_index.items():
    if word in word2vec.wv.vocab:
        embedding_matrix[i] = word2vec.wv.word_vec(word)
    else:
        #print word,
        pass
        #embedding_matrix[i] = np.random.randn(EMBEDDING_DIM)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

#np.save(embedding_matrix_path, embedding_matrix)

# sequence_length = x.shape[1]
# vocabulary_size = len(vocabulary_inv)
# print "sequence len: ", sequence_length
# print "vocab size: ", vocabulary_size
args.filter_sizes = [3, 4, 5, 6]
args.nb_words = nb_words
args.embedding_matrix = embedding_matrix
#args.embedding_matrix = None 

model = get_model(args)
if args.snapshot:
    model.load_weights(args.snapshot)
if args.test:
    test_set = []
    int_labels_test = []
    with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
        for line in f:
            # print  line
            tokens = line.strip().split(",")
            try:
                int_labels_test.append(label_map[tokens[2]])
                test_set.append(tokens[1].lower())
            except Exception, e:
                print "--------------", e.message
    sequences_test = tokenizer.texts_to_sequences(test_set)
    test_data = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
    print "start predict....."
    ans = np.argmax(model.predict(test_data), axis=1)
    print "save ans npy...."
    np.save("../checkpoint/result.npy", ans)
    print "save ans to submit...."

    print "accuracy: ", accuracy_score(int_labels_test, ans)
    print "F1: ", f1_score(int_labels_test, ans, average=None)
    cnf_matrix = confusion_matrix(int_labels_test, ans)
    print cnf_matrix
elif args.predict:
    pass
else:
    self_train(labeled_train_set, train_set_labels, unlabel_set, model, args)
