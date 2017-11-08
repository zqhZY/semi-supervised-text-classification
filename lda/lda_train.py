# -*- encoding:utf-8 -*-
import random
import sys
import codecs
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import numpy as np

reload(sys)
sys.setdefaultencoding("utf-8")

train_file = "/home/zqh/mygit/cincc/semanaly/tools/data/mobile_dataset_jieba.csv"
test_file = "/home/zqh/mygit/cincc/semanaly/tools/data/mobile_dataset_jieba_test_cleaned.csv"
corpus = []
lables = []
stopwords = codecs.open('stop_words_ch.txt', 'r', encoding='utf8').readlines()
stopwords = [w.strip() for w in stopwords]
# print stopwords

label_map = {u"办理":0, u"投诉（含抱怨）":1, u"咨询（含查询）":2, u"其他":3, u"表扬及建议":4}
train_corpus = []
with codecs.open(train_file, 'r', encoding='utf8') as f:
    first_line = True
    for line in f:
        if first_line:
            first_line = False
            continue
        tokens = line.strip().split(",")
        train_corpus.append([tokens[1], label_map[tokens[2]]])
        # print tokens[2]
        # print line
        # lables.append(label_map[tokens[2]])

test_corpus = []
ans = []
with codecs.open(test_file, 'r', encoding='utf8') as f:
    first_line = True
    for line in f:
        if first_line:
            first_line = False
            continue
        tokens = line.strip().split(",")
        test_corpus.append(tokens[1])
        ans.append(label_map[tokens[2]])

random.shuffle(train_corpus)
train_text = []
lables = []
for t, l in train_corpus:
    train_text.append(t)
    lables.append(l)

corpus = train_text + test_corpus

print len(train_text)
print len(test_corpus)
print len(corpus)
print lables
print ans
cntVector = CountVectorizer(stop_words=stopwords)
cntTf = cntVector.fit_transform(corpus)
print len(cntVector.get_feature_names())
np.save("snapshots/count.npy", cntTf)


lda = LatentDirichletAllocation(n_topics=50,
                                learning_offset=50.,
                                random_state=0)
docres = lda.fit_transform(cntTf)

np.save("snapshots/lda_docres.npy", docres)
print type(docres)


train_features = docres[0:len(train_corpus)]
test_features = docres[len(train_corpus):]

## rf train
clf = RandomForestClassifier(max_depth=6, random_state=0)
clf.fit(train_features, lables)

## svm train
classifier = svm.SVC(kernel='linear', C=0.01)
classifier.fit(train_features, lables)


import pickle
pickle.dump(clf, open("snapshots/rf.mod", "w"))

obj = pickle.load(open("snapshots/rf.mod"))
print(obj.predict(test_features))