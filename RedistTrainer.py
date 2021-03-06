# read input data
import os
import tensorflow as tf
import numpy as np

xy_lst = []
dirc = ".\\data\\"
while True:
    try:
        folder = input(" entry a data folder: ")
        path = dirc + folder
        for file in os.listdir(path):
            filename = path + "\\" + file
            print(filename)
            f = open(filename)
            for line in f:
                words = line.split(' ')
                xy_lst.append(list(map(float, words)))
            f.close()
    except IOError:
        print(IOError)
        break

xy = np.array(xy_lst)
np.random.shuffle(xy)

x_data = xy[:,:-2]
y_data = xy[:,-2:]

layers = [48, 24, 8, 4, 2]
maxIter = 100000
alpha = 0.03
reg = 0.0
batch_size = 1000

x = tf.placeholder(tf.float32, shape=[None, layers[0]], name="x")
y_ = tf.placeholder(tf.float32, shape=[None, layers[-1]], name="y_")

a, w, b = [x], [], []
for i in range(1, len(layers)):
    shape = (layers[i-1], layers[i])
    w.append(tf.Variable(tf.random_normal(shape, stddev=0.1), name="w"+str(i)))
    b.append(tf.Variable(tf.zeros(layers[i]), name="b"+str(i)))
    a.append(tf.placeholder(tf.float32, shape=[None, layers[i]], name="a"+str(i)))

for i in range(1, len(layers) - 1):
    a[i] = tf.nn.relu(tf.matmul(a[i-1], w[i-1]) + b[i-1])
y = tf.matmul(a[-2], w[-1]) + b[-1]
y = tf.identity(y, name="y")

cost = tf.reduce_mean(tf.square(y - y_, name="cost"))
train = tf.train.GradientDescentOptimizer(alpha).minimize(cost, name="train_op")

n = y_data.shape[0]
n_train = int(0.7* n)
n_batch = n_train // batch_size
x_train = x_data[:n_train,:]
y_train = y_data[:n_train,:]
x_val = x_data[n_train:,:]
y_val = y_data[n_train:,:]

saver = tf.train.Saver()
#start training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(maxIter):
        for i in range(n_batch):
            start = i* batch_size
            stop = np.minimum(start+batch_size, n_train)
            train.run(feed_dict={x:x_train[start:stop,:], y_:y_train[start:stop,:]})
        print(epoch, "train cost: ", cost.eval(feed_dict={x:x_train, y_:y_train}))
        print(epoch, "test cost: ", cost.eval(feed_dict={x:x_val, y_:y_val}))
        if epoch%1000 == 1:
            saver.save(sess, "./model")
