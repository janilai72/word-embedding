# -*- encoding: utf-8 -*-
'''
@File    :   find_word.py
@Time    :   2018/12/01 17:53:15
@Author  :   Jian.Lai
@Version :   1.0
@Contact :   jani.lai@aliyun.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   None
'''
import os
import jieba
from new_word_model import TrieNode
from utils import get_stopwords, load_dictionary, generate_ngram, save_model, load_model


def load_data(filename, stopwords):
    """
    :param filename:
    :param stopwords:
    :return: 二维数组,[[句子1分词list], [句子2分词list],...,[句子n分词list]]
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            word_list = [x for x in jieba.cut(line.strip(), cut_all=False) if x not in stopwords]
            data.append(word_list)
    return data


def load_data_2_root(data):
    print('------> 插入节点')
    for word_list in data:
        # tmp 表示每一行自由组合后的结果（n gram）
        # 雪落/ 山庄/ 不是/ 一座/ 山庄/ 只是/ 一个/ 客栈
        # tmp: [['雪落'], ['山庄'], ['不是'], ['一座'],['山庄'],['只是'],['一个'],['客栈'],
        # ['雪落', '山庄'], ['山庄', '不是'], ['不是', '一座'],['一座','山庄'],['山庄','只是'],['只是','一个'],['一个','客栈']
        #  ['雪落', '山庄', '不是'], ['山庄', '不是', '一座'],[不是'','一座','山庄'],...
        ngrams = generate_ngram(word_list, 3)
        #print(ngrams)
        for d in ngrams:
            root.add(d)
    print('------> 插入成功')
                                                                                            

if __name__ == "__main__":

    root = TrieNode('*', None)
    stopwords = get_stopwords('./data/stopword.txt')
    data = load_data('./data/data.txt', stopwords)
    # 将新的文章插入到Root中
    load_data_2_root(data)

    # 定义取TOP5个
    topN = 20
    result, add_word = root.find_word(topN)
    # 如果想要调试和选择其他的阈值，可以print result来调整
    # print("\n----\n", result)
    print("\n----\n", '增加了 %d 个新词, 词语和得分分别为: \n' % len(add_word))
    print('#############################')
    for word, score in add_word.items():
        print(word + ' ---->  ', score)
    print('#############################')

    # 前后效果对比
    test_sentence = '雪落山庄不是一座山庄，只是一个客栈，还是个很破很破的客栈，方圆百里也只有这一家客栈。它背靠一座高山，面朝一条大河。翻越那座山需要很久，越过那条河也并不容易，所以成了赶路人中途歇息的必选之地。'
    print('添加前：')
    print("".join([(x + '/ ') for x in jieba.cut(test_sentence, cut_all=False) if x not in stopwords]))

    for word in add_word.keys():
        jieba.add_word(word)
    print("添加后：")
    print("".join([(x + '/ ') for x in jieba.cut(test_sentence, cut_all=False) if x not in stopwords]))