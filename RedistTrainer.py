# read input data
import tensorflow as tf
import numpy as np

xy_lst = []
dirc = ".\\data\\"
while True:
    try:
        filename = input(" data file to read: ")
        path = dirc + filename
        print path
        f = open(path)
        for line in f:
            words = line.split(' ')
            xy_lst.append(map(float, words))
        f.close()
    except IOError:
        print IOError
        break

xy = np.array(xy_lst)
np.random.shuffle(xy)

x_data = xy[:,:-2]
y_data = xy[:,-2:]

layers = [16, 8, 8, 2]
maxIter = 5000
alpha = 0.03
reg = 0.0
batch_size = 200

x = tf.placeholder(tf.float32, shape=[None, layers[0]])
y_ = tf.placeholder(tf.float32, shape=[None, layers[-1]])

a, w, b = [], [], []
for i in range(1, len(layers)):
    shape = (layers[i-1], layers[i])
    w.append(tf.Variable(tf.random_normal(shape, stddev=0.1)))
    b.append(tf.Variable(tf.zeros(layers[i])))
    a.append(tf.Variable(tf.zeros(layers[i])))

y = tf.Variable(tf.zeros(layers[-1]))
for in range(1, len(layers)):
    a[i] = tf.nn.relu(tf.matmul(x, w) + b)
y = a[-1]

cost = tf.norm(y - y_, ord=2)
train = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

#start training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    n = y_data.shape[0]
    n_train = int(0.7* n)
    n_batch = n_train / batch_size
    x_train = x_data[:n_train,:]
    y_train = y_data[:n_train,:]
    x_val = x_data[n_train:,:]
    y_val = y_data[n_train:,:]
    for epoch in range(maxIter):
        for i in range(n_batch):
            start = i* batch_size
            stop = np.minimum(start+batch_size, n_train)
            train.run(feed_dict={x:x_train[start:stop,:], y_:y_train[start:stop,:]})
        print("cost: ", cost)
