import tensorflow as tf
#tensorflow.org
#placeholder 생성 graph를 생성하지 않는다. input date를 쓸때
placeholder = tf.placeholder(tf.float32, shape=[3,3])

#variables 하나의 객체
variables = tf.Variable([1,2,3,4,5,6],dtype=tf.float32)

#constant 상수 고정 값을 지정
constant = tf.constant([10,20,30,40,50],dtype=tf.float32)

#tensorflow 자체적으로 print불가 run으로 해야 값 확인 가능
# sess = tf.Session()
# result = sess.run(constant)
# print(result)

a = tf.constant([5])
b = tf.constant([10])
c = tf.constant([2])

d = a + b + c
#session에 올라가야 연산이 가능
sess = tf.Session()
#run을 활용하여 연산과정 진행
result = sess.run(d)
print(result)

#key , value 형태로
value1 = 5
value2 = 3
value3 = 2

ph1 = tf.placeholder(dtype=tf.float32)
ph2 = tf.placeholder(dtype=tf.float32)
ph3 = tf.placeholder(dtype=tf.float32)

rs_value = ph1 + ph2 + ph3
#dict형태로 만들기
feed_dict = {ph1:value1,ph2:value2,ph3:value3}
result2 = sess.run(rs_value,feed_dict=feed_dict)
print(result2)


image = [1,2,3,4,5]
label = [10,20,30,40,50]

ph_image = tf.placeholder(dtype=tf.float32)
ph_label = tf.placeholder(dtype=tf.float32)

feed_dict1 = {ph_image: image, ph_label:label}

result_tf = ph_image + ph_label
result3 = sess.run(result_tf, feed_dict=feed_dict1)
print(result3)


