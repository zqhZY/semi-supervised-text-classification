# -*- coding:utf-8 -*-
import re
import string
import sys

from gensim.models import Word2Vec

reload(sys)
sys.setdefaultencoding("utf-8")


# rm url, punc
re_url = re.compile(r'http://[a-zA-Z0-9.?/&=:]*', re.S)
punc = string.punctuation + "“”：《》～，。/‘；’】【}{+=）（……！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
re_punc = re.compile(ur"[%s]+" % punc, re.S)


EMBEDDING_FILE = 'word2vector/model/news_data_w2v_min5_mod'

print('Preparing embedding matrix')
word2vec = Word2Vec.load(EMBEDDING_FILE)
print len(word2vec.wv.vocab)
with open("data/news_data/news_data.txt") as f, open("news_data_train_rm_low_freq.csv", "w") as f2:
    skip = True
    f2.write("id,text,label\n")
    for line in f:
        if skip:
            skip = False
            continue
        tokens = line.decode("utf-8").strip().split(",")
        print tokens[1]
        text = re_url.sub("", tokens[1])
        text = re_punc.sub("", text.decode("utf-8"))
        words = text.split(" ")
        tmp_text = []
        for word in words:
            if word in word2vec.wv.vocab:
                tmp_text.append(word)
        f2.write(",".join([tokens[0], " ".join(words), tokens[2]]) + "\n")
