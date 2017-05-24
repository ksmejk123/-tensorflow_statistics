import tensorflow as tf
import numpy as np

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
#None 2차원이기 떄문에 input_size [[]]안에 넣어서 실행
x = tf.placeholder(tf.float32, shape=[None,INPUT_SIZE])
y = tf.placeholder(tf.float32, shape=[None,CLASSES])

feed_dict = {x: input_data, y:lable_data}
#truncated_normal
w_h1 = tf.Variable(tf.truncated_normal(shape=[INPUT_SIZE,HIDDEN1_SIZE],dtype=tf.float32))
#shpae은 []안에 들어가게
b_h1 = tf.Variable(tf.zeros([HIDDEN1_SIZE]),dtype=tf.float32)
#메트릭스 곱은 다른형식이기 때문에 tensor 사용
hidden1 = tf.sigmoid(tf.matmul(x ,w_h1) + b_h1)

#hidden2 layer 부분
w_h2 = tf.Variable(tf.truncated_normal(shape=[HIDDEN1_SIZE,HIDDEN2_SIZE],dtype=tf.float32))
b_h2 = tf.Variable(tf.zeros([HIDDEN2_SIZE]),dtype=tf.float32)

hidden2 = tf.sigmoid(tf.matmul(hidden1, w_h2) + b_h2)
#output Variable 구성
w_o = tf.Variable(tf.truncated_normal(shape=[HIDDEN2_SIZE,CLASSES],dtype=tf.float32))
b_o = tf.Variable(tf.zeros([CLASSES]),dtype=tf.float32)
#모델 부분
#out_put 부분은 add연산 노드가 추가
out_put = tf.sigmoid(tf.matmul(hidden2,w_o) + b_o)

#cost function
cost = tf.reduce_mean(-y * tf.log(out_put) - (1-y)*tf.log(1-out_put))
train = tf.train.GradientDescentOptimizer(Learning_Rate).minimize(cost)
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
for i in range(1000):
    __, loss = sess.run([train,cost],feed_dict=feed_dict)
    if i % 100 == 0:
        print("Step:",loss)
sess.close()




