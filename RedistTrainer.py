import tensorflow as tf
sess = tf.InteractiveSession()

layers = [16, 8, 8, 2]

x = tf.placeholder(tf.float32, shape=[None, layers[0]])
y_ = tf.placeholder(tf.float32, shape=[None, layers[-1]])

w, b = [], []
for i in range(1, len(layers)):
    shape = (layers[i], layers[i-1])
    w.append(tf.Variable(tf.random_normal(shape, stddev=0.1)))
    b.append(tf.Variable(tf.zeros(layers[i])))

sess.run(tf.global_variables_initializer())

y = tf.Variable(tf.zeros(layers[-1]))
