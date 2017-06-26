import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

#Prediction Model 코드

input_data = [[1,5,3,7,8,10,12],
              [5,8,10,3,9,7,1]]
lable_data = [[0,0,0,1,0],
              [1,0,0,0,0]]
#shape은 차원크기와 갯수로
INPUT_SIZE = 7
HIDDEN1_SIZE = 10
HIDDEN2_SIZE = 8
CLASSES= 5
Learning_Rate = 0.05

x = tf.placeholder(tf.float32, shape=[None,INPUT_SIZE],name='x')
y = tf.placeholder(tf.float32, shape=[None,CLASSES],name='y')

feed_dict = {x: input_data, y:lable_data}

#hidden2 layer 부분
w_h1 = tf.Variable(tf.truncated_normal(shape=[INPUT_SIZE,HIDDEN1_SIZE],dtype=tf.float32),name='w_h1')
b_h1 = tf.Variable(tf.zeros([HIDDEN1_SIZE]),dtype=tf.float32,name='b_h1')
#hidden2 layer 부분
w_h2 = tf.Variable(tf.truncated_normal(shape=[HIDDEN1_SIZE,HIDDEN2_SIZE],dtype=tf.float32),name='w_h2')
b_h2 = tf.Variable(tf.zeros([HIDDEN2_SIZE]),dtype=tf.float32,name='b_h2')
#output Variable 구성
w_o = tf.Variable(tf.truncated_normal(shape=[HIDDEN2_SIZE,CLASSES],dtype=tf.float32),name='w_o')
b_o = tf.Variable(tf.zeros([CLASSES]),dtype=tf.float32,name='b_o')

param_list = [w_h1,b_h1,w_h2,b_h2,w_o,b_o]
saver = tf.train.Saver(param_list)

#형태 부분
hidden1 = tf.sigmoid(tf.matmul(x ,w_h1) + b_h1,name='hidden1')
hidden2 = tf.sigmoid(tf.matmul(hidden1, w_h2) + b_h2,name='hidden2')
out_put = tf.sigmoid(tf.matmul(hidden2,w_o) + b_o,name='out_put')

sess = tf.Session()
#Initialization 부분에서 tarin파일 load
# sess.run(tf.global_variables_initializer())

#저장 된 tarin파일 불러오는 부분 Initialization
saver.restore(sess, './tensorflow_chkpt.ckpt')
result = sess.run(out_put, feed_dict)
print(result)




