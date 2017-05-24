import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_points = 200
vectors_set = []
for i in range(num_points):
    #난수 생성 5 중심으로 표준편차가 5인데이터에 +15
    x = np.random.normal(5,5) + 15
    #x(거리) * 1000 + 난수 생성 0 중심으로 표준편차가 3인데이터 + 1000
    y = x * 1000 + (np.random.normal(0,3)) * 1000
    vectors_set.append([x,y])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.ProximalGradientDescentOptimizer(0.0015)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in range(10):
    sess.run(train)
    print(step,sess.run(W),sess.run(b))
    print(step,sess.run(loss))

