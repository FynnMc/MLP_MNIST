import tensorflow as tf
import numpy as np
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data

now = datetime.now()


def accuracy(label, pred):
    return np.mean(np.argmax(pred, axis=1) == np.argmax(label, axis=1))*100.0


def simple_fnn():

    # Hyperparams and Loadin

    data = input_data.read_data_sets("MNIST_Data/", one_hot=True)
    logdir = "C:\\Users\\Fynn\\PycharmProjects\\MLP\\Summary\\" + now.strftime("%Y%m%d-%H%M%S")

    layer_ids = ['hidden1', 'hidden2', 'hidden3', 'output']
    layer_size = [784, 500, 250, 100, 10]

    batch_size = 100
    epochs = 25
    eta = 0.01
    mom = 0.9

    n_train = 55000
    n_valid = 5000
    n_test = 10000

    # GRAPH
    # tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():

        # Initial Placeholders
        x_input = tf.placeholder(dtype=tf.float32, shape=[None, layer_size[0]], name='input')
        y_labels = tf.placeholder(dtype=tf.float32, shape=[None, layer_size[-1]], name='output')

        # Weights/Biases
        for ind, name in enumerate(layer_ids):
            with tf.variable_scope(name):
                w = tf.get_variable(name='weight', shape=[layer_size[ind], layer_size[ind+1]], dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer()) # Final Out of Range?
                b = tf.get_variable(name='bias', shape=layer_size[ind+1], dtype=tf.float32,
                                    initializer=tf.initializers.random_uniform(-0.1, 0.1))

        # Hidden Layers
        h = x_input
        for name in layer_ids:
            with tf.variable_scope(name, reuse=True):
                w = tf.get_variable('weight')
                b = tf.get_variable('bias')
                if name != 'output':
                    h = tf.nn.relu(tf.matmul(h, w) + b, name=name+'_output')
                else:
                    h = tf.nn.xw_plus_b(h, w, b, name=name + '_output')

        # Output Layer and Loss
        tf_pred = tf.nn.softmax(h, name='predictions')
        tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_labels, logits=h), name='loss')

        # Optimisation
        optimiser = tf.train.MomentumOptimizer(learning_rate=eta, momentum=mom)
        grad_var = optimiser.compute_gradients(loss=tf_loss)
        tf_loss_minimise = optimiser.minimize(tf_loss)

        # Scalar Summary
        with tf.name_scope('performance'):
            tf_loss_ph = tf.placeholder(dtype=tf.float32, shape=None, name='loss_summary')
            tf_tr_acc_ph = tf.placeholder(dtype=tf.float32, shape=None, name='train_acc_summary')
            tf_v_acc_ph = tf.placeholder(dtype=tf.float32, shape=None, name='valid_acc_summary')

            tf_loss_summary = tf.summary.scalar('loss', tf_loss_ph)
            tf_tr_acc_summary = tf.summary.scalar('tr_acc', tf_tr_acc_ph)
            tf_v_acc_summary = tf.summary.scalar('v_acc', tf_v_acc_ph)

        for gr, v in grad_var:
            if 'hidden3' in v.name and 'weight' in v.name:
                with tf.name_scope('gradient'):
                    tf_last_grad_norm = tf.sqrt(tf.reduce_mean(gr ** 2))
                    tf_gradnorm_summary = tf.summary.scalar('grad_norm', tf_last_grad_norm)
                    break

        # Dist Summary
        layer_summary = []
        for lid in layer_ids:
            with tf.name_scope(lid + '_hist'):
                with tf.variable_scope(lid, reuse=True):
                    w = tf.get_variable('weight')
                    b = tf.get_variable('bias')
                    tf_w_hist = tf.summary.histogram('weight_hist', tf.reshape(w, [-1]))
                    tf_b_hist = tf.summary.histogram('bias_hist', b)
                    layer_summary.extend([tf_w_hist, tf_b_hist])
        tf_dist_wb = tf.summary.merge(layer_summary)

        # Summary Merge
        writer = tf.summary.FileWriter(logdir)
        perf_summaries = tf.summary.merge([tf_loss_summary, tf_tr_acc_summary, tf_v_acc_summary])

        # Variable Init
        init_op = tf.global_variables_initializer()

    # SESSION
    with tf.Session(graph=g) as sess:
        sess.run(init_op)
        for epoch in range(epochs):
            # ========== Train ==========
            loss_per_epoch = []
            for batch in range(n_train // batch_size):
                train_b_x, train_b_y = data.train.next_batch(batch_size)
                if batch == 0:
                    l, _, gn_summary, wb_summary = sess.run([tf_loss, tf_loss_minimise, tf_gradnorm_summary,
                                                             tf_dist_wb], feed_dict={x_input: train_b_x,
                                                                                     y_labels: train_b_y})
                    writer.add_summary(gn_summary, epoch)
                    writer.add_summary(wb_summary, epoch)
                else:
                    l, _ = sess.run([tf_loss, tf_loss_minimise], feed_dict={x_input: train_b_x,
                                                                            y_labels: train_b_y})
                loss_per_epoch.append(l)
            avg_tr_loss = np.mean(loss_per_epoch)
            train_x, train_y = data.train.next_batch(batch_size)
            tr_pred = sess.run(tf_pred, feed_dict={x_input: train_x, y_labels: train_y})
            tr_acc = accuracy(train_y, tr_pred)
            print('Average Training Loss in Epoch {}: {}'.format(epoch+1, avg_tr_loss))
            print('\tTraining Accuracy in Epoch {}: {}'.format(epoch+1, tr_acc))
            # ========== Valid ==========
            valid_b_x, valid_b_y = data.train.next_batch(n_valid)
            v_pred = sess.run(tf_pred, feed_dict={x_input: valid_b_x, y_labels: valid_b_y})
            v_acc = accuracy(valid_b_y, v_pred)
            print('\tValidation Accuracy in Epoch {}: {}'.format(epoch+1, v_acc))
            # =========== Summary =========
            summary = sess.run(perf_summaries, feed_dict={tf_loss_ph: avg_tr_loss, tf_tr_acc_ph: tr_acc,
                                                          tf_v_acc_ph: v_acc})
            writer.add_summary(summary, epoch)

        writer.add_graph(sess.graph)


simple_fnn()
