#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   text_pre.py
@Time    :   2019/01/18 16:45:06
@Author  :   Jian.Lai
@Version :   1.0
@Contact :   jani.lai@aliyun.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   None
'''


import jieba
import re
jieba.load_userdict("./data/dict.txt")


re_url = re.compile(u"((ht|f)tps?):\/\/([\w\-]+(\.[\w\-]+)*\/)*[\w\-]+(\.[\w\-]+)*\/?(\?([\w\-\.,@?^=%&:\/~\+#]*)+)?")
re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z]+)")


stopwords = []
with open('./data/stopword.txt','r') as f:
    for line in f.readlines():
        stopwords.append(line.strip())

content = []
with open('./data/shaoniangexing.txt','r',encoding='UTF-8-sig') as f:
    
    for line in f.readlines():
        line = re_url.sub('',line).strip()
        arr = re_han.split(line)
        for sw in arr:
            if sw.strip() in stopwords:
                continue
            content.extend(jieba.lcut(sw))

with open('./data/shaoniangexing_seg.txt','w',encoding='utf-8') as fw:
    fw.write(' '.join(content))




        