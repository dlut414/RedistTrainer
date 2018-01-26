import tensorflow as tf
import numpy as np
xx = np.random.random((200,48))* 2 - 1
with tf.Session() as sess:
    saver = tf.train.import_meta_graph("./model.meta")
    saver.restore(sess, tf.train.latest_checkpoint("./"))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    a2 = graph.get_tensor_by_name("a2:0")
    # print(a2.eval(feed_dict={x:xx}))
    print(y.eval(feed_dict={x:xx}))
    # print(x.eval(feed_dict={x:xx}))
