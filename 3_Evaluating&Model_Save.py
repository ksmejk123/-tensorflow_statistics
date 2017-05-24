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

#hidden2 layer 부분
w_h1 = tf.Variable(tf.truncated_normal(shape=[INPUT_SIZE,HIDDEN1_SIZE],dtype=tf.float32))
b_h1 = tf.Variable(tf.zeros([HIDDEN1_SIZE]),dtype=tf.float32)
#hidden2 layer 부분
w_h2 = tf.Variable(tf.truncated_normal(shape=[HIDDEN1_SIZE,HIDDEN2_SIZE],dtype=tf.float32))
b_h2 = tf.Variable(tf.zeros([HIDDEN2_SIZE]),dtype=tf.float32)
#output Variable 구성
w_o = tf.Variable(tf.truncated_normal(shape=[HIDDEN2_SIZE,CLASSES],dtype=tf.float32))
b_o = tf.Variable(tf.zeros([CLASSES]),dtype=tf.float32)

#save point 들어가는 부분
param_list = [w_h1,b_h1,w_h2,b_h2,w_o,b_o]
saver = tf.train.Saver(param_list)


hidden1 = tf.sigmoid(tf.matmul(x ,w_h1) + b_h1)
hidden2 = tf.sigmoid(tf.matmul(hidden1, w_h2) + b_h2)
#out_put 부분은 add연산 노드가 추가
out_put = tf.sigmoid(tf.matmul(hidden2,w_o) + b_o)



#cost function
cost = tf.reduce_sum(-y * tf.log(out_put) - (1-y)*tf.log(1-out_put))
cost = tf.reduce_mean(cost)
train = tf.train.GradientDescentOptimizer(Learning_Rate).minimize(cost)

#accuracy 부분
comp_pred = tf.equal(tf.argmax(out_put,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(comp_pred, tf.float32))

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(1000):
    __, loss,acc = sess.run([train,cost,accuracy], feed_dict)

    if i % 100 == 0:
        # Saves 저장
        saver.save(sess,'./tensorflow_chkpt.ckpt')
        print('step:',i)
        print('loss',loss)
        print('accracy',acc)
sess.close()
#tf.equal은 max index값이 같은지 확인 bool형







