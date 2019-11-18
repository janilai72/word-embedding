from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
##加载目录下的文件 数据量大于内存空间
#sentences = word2vec.PathLineSentences("./data/seg") 
#加载单个文件 处理小文件
sentences = word2vec.LineSentence('./data/shaoniangexing_seg.txt') 



"""
    size:词向量
    sg:训练模型 0：CBOW 1：Skip-Gram
    hs:优化算法 0：Negative Sampling 1：Hierarchical Softmax
    window:词向量上下文
    min_count:最小词频
    workers:工作线程
    iter:迭代的最大次数
"""
#model = word2vec.Word2Vec(sentences,sg=1,size=50,hs=0, window=5, min_count=5, workers=8,iter=5)
#model.save("./model/gensim/xiaoshuo.model")
# model.wv.save_word2vec_format("./model/gensim/xiaoshuo.txt", binary=False)
# model.wv.save_word2vec_format("./model/gensim/xiaoshuo.bin", binary=True)

model=word2vec.Word2Vec.load("./model/gensim/xiaoshuo.model")
#近似的词
print(model.most_similar('雷无桀'))
print(model.most_similar('无极棍'))


print(model.wv.similarity('天外天','天启城'))

