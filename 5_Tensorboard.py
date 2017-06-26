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

tensor_map = {x: input_data, y:lable_data}

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

with tf.name_scope('hidden_layer_1') as h1scope:
    hidden1 = tf.sigmoid(tf.matmul(x ,w_h1) + b_h1,name='hidden1')
with tf.name_scope('hidden_layer_2') as h2scope:
    hidden2 = tf.sigmoid(tf.matmul(hidden1, w_h2) + b_h2,name='hidden2')
with tf.name_scope('outpur_layer') as oscope:
    out_put = tf.sigmoid(tf.matmul(hidden2,w_o) + b_o,name='out_put')

with tf.name_scope('calculate_costs'):
    cost = tf.reduce_sum(-y * tf.log(out_put) - (1-y)*tf.log(1-out_put))
    cost = tf.reduce_mean(cost)
    tf.summary.scalar('cost/',cost)
with tf.name_scope('trianing'):
    train = tf.train.GradientDescentOptimizer(Learning_Rate).minimize(cost)
with tf.name_scope('evaluation'):
    comp_pred = tf.equal(tf.argmax(out_put,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(comp_pred, tf.float32))

#summary는 log를 모으는 형태
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

merge = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(logdir='./summaries/',graph=sess.graph)

for i in range(1000):
    summary, _, loss, acc = sess.run([merge,train,cost,accuracy],tensor_map)
    if i % 100 == 0:
        train_writer.add_summary(summary,i)
        saver.save(sess, './tensorflow_chkpt.ckpt')
        print('-------------')
        print('step:',i)
        print('loss',loss)
        print('accuracy',acc)
sess.close()

#graph 파일 인식을 못해서 tensorboard에서 graph파일 확인 불가
#Embeddings 확인가능




