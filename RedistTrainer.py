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

sess = tf.InteractiveSession()

layers = [16, 8, 8, 2]

x = tf.placeholder(tf.float32, shape=[None, layers[0]])
y_ = tf.placeholder(tf.float32, shape=[None, layers[-1]])

a, w, b = [], [], []
for i in range(1, len(layers)):
    shape = (layers[i-1], layers[i])
    w.append(tf.Variable(tf.random_normal(shape, stddev=0.1)))
    b.append(tf.Variable(tf.zeros(layers[i])))
    a.append(tf.Variable(tf.zeros(layers[i])))

sess.run(tf.global_variables_initializer())

y = tf.Variable(tf.zeros(layers[-1]))
for in range(1, len(layers)):
    a[i] = tf.nn.relu(tf.matmul(x, w) + b)
y = a[-1]

cost = tf.norm(y - y_, ord=2)
train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

#start training

