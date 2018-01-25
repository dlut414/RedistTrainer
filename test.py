import tensorflow as tf
import numpy as np
saver = tf.train.Saver()
x = tf.get_variable("x")
y = tf.get_variable("y")
xx = np.ones((1,48))
with tf.Session() as sess:
    saver.restore(sess, "./model.ckpt")
    print(y.eval(feed_dict={x:xx}))
