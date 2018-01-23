import tensorflow as tf
sess = tf.InteractiveSession()

layers = [16, 8, 8, 2]

x = tf.placeholder(tf.float32, shape=[None, layers[0]])
y = tf.placeholder(tf.float32, shape=[None, layers[-1]])

w = tf.Variable(tf.random_normal(layers))
b = tf.Variable(tf.zeros(layers[1:]))

sess.run(tf.global_variables_initializer())


