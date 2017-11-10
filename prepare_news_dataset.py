# -*- coding:utf-8 -*-
import re
from glob import glob
from tqdm import tqdm
import jieba
import sys

reload(sys)
sys.setdefaultencoding("utf-8")

data_set = {}
punc = u")(<>./;'`!@#$%^&*()_+《》！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
punc = punc.decode("utf-8")

dir_list = sorted(glob("./data/news_data/*"))
with open("data/news_data/news_data.txt", "w") as f:
    f.write("id,text,label\n")
    idx = 0
    for class_dir in dir_list:
        class_name = class_dir.split("/")[3]
        file_list = sorted(glob(class_dir + "/*"))
        for record in tqdm(file_list):
            with open(record) as f1:
                text = ""
                for line in f1:
                    text += line.decode("utf-8").strip().lstrip().replace(" ", "").replace(",", "，")
                # text = f1.read().decode("utf-8").replace("\n", "")
                words = jieba.cut(text, cut_all=False)
                f.write(str(idx) + "," + " ".join(words).lower() + "," + "\n")
                idx += 1

