import tensorflow as tf
import numpy as np
xx = np.random.random((1,48))* 10
with tf.Session() as sess:
    saver = tf.train.import_meta_graph("./model.meta")
    saver.restore(sess, tf.train.latest_checkpoint("./"))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    output = sess.run(y, feed_dict={x:xx})
    print(output)
