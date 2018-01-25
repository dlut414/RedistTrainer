import tensorflow as tf
import numpy as np
x = tf.get_variable("x", shape=[1, 48])
y = tf.get_variable("y", shape=[1, 2])
saver = tf.train.Saver()
xx = np.ones((1,48))
with tf.Session() as sess:
    saver.restore(sess, "./model.ckpt")
    print(y.eval(feed_dict={x:xx}))
