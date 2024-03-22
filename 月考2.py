# 2.0深度学习是以神经网络为基础的，请使用相应算法实现数据集books中的多分类任务(最后列为类别，其余为特征列)：
# (1)引入相关模块包(2分)
import numpy as np
import tensorflow as tf
import random
tf.set_random_seed(888)
# (2)利用numpy中的loadtxt读取数据文件(2分)
data=np.loadtxt('books.csv',delimiter=',')
# (3)提取特征与标签并切分训练集与测试集(2分)
from sklearn.model_selection import train_test_split
x_data=data[:,:-1]
y_data=data[:,-1]
train_x,test_x,train_y,test_y=train_test_split(x_data,y_data)
# (4)自定义小批量函数(2分)
def next_batch(batch_size):
    global point
    batch_x=train_x[point:point+batch_size]
    batch_y=train_y[point:point+batch_size]
    point+=batch_size
    return batch_x,batch_y
# (5)定义占位符张量X,float32类型；(2分)
X=tf.placeholder(dtype=tf.float32,shape=[None,x_data.shape[1]])
# (6)定义占位符张量Y，int32类型 (2分)
Y=tf.placeholder(dtype=tf.int32,shape=[None,])
# (7)把Y进行one-hot操作 (2分)
y_one_hot=tf.one_hot(Y,len(set(y_data)))
# (8)设置第一层参数w1,设置参数b1(2分)
w1=tf.Variable(tf.random_normal([x_data.shape[1],64]))
b1=tf.Variable(tf.random_normal([64]))
# (9)利用sigmoid函数激活(2分)
h1=tf.sigmoid(tf.matmul(X,w1)+b1)
# (10)设置第二层参数w2,设置参数b2(2分)
w2=tf.Variable(tf.random_normal([64,len(set(y_data))]))
b2=tf.Variable(tf.random_normal([len(set(y_data))]))
# (11)定义预测函数(2分)
h=tf.matmul(h1,w2)+b2
# (12)定义代价函数loss(2分)
loss=tf.nn.softmax_cross_entropy_with_logits(logits=h,labels=y_one_hot)
# (13)定义准确率(2分)
y_true=tf.argmax(h,-1)
y_predict=tf.argmax(y_one_hot,-1)
acc=tf.reduce_mean(tf.cast(tf.equal(y_predict,y_true),tf.float32))
# (14)使用优化器实现梯度下降 (2分)
op=tf.train.AdamOptimizer(0.01).minimize(loss)
# (15)创建Session(2分)
# (16)全局变量初始化(2分)
# (17)自设置大循环次数epoch=10(2分)
# (18)小批量大小为50batch_size=50(2分)
# (19)计算总批次数(2分)
# (20)使用训练集的数据进行小批量训练(2分)
# (21)每100次输出一次cost值(2分)
# (22)计算测试集准确率(2分)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epoch=10
    batch_size=50
    batch=len(train_y)//batch_size
    for i in range(epoch):
        point=0
        avg_loss=0
        for j in range(batch):
            batch_x,batch_y=next_batch(batch_size)
            loss_,op_=sess.run([loss,op],feed_dict={X:batch_x,Y:batch_y})
            avg_loss+=loss_
        print(i,avg_loss/batch)
    print(sess.run(acc,feed_dict={X:test_x,Y:test_y}))
