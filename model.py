import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import numerics

tf.app.flags.DEFINE_integer(
    'batch_size', 256, 'The batch size used to train the model')
tf.app.flags.DEFINE_integer(
    'epoch_size', 10, 'The number of epo of training')
tf.app.flags.DEFINE_string(
    'entity_feature_path', './data/ent_embeddings_transE.txt', 'The file stored entity features')
tf.app.flags.DEFINE_string(
    'relation_feature_path', './data/rel_embeddings_transE.txt', 'The file stored relation features')
tf.app.flags.DEFINE_string(
    'train_path', './data/triple2id.txt', 'The triple2id dataset of training')
tf.app.flags.DEFINE_string(
    'valid_path', './data/valid2id.txt', 'The triple2id dataset of validation')
tf.app.flags.DEFINE_string(
    'test_path', './data/test2id.txt', 'The triple2id dataset of test')
tf.app.flags.DEFINE_string(
    'log_path', './log_model_1', 'The log path')
tf.app.flags.DEFINE_string(
    'checkpoint_path', './log/model_6.ckpt',
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'device_mode', 'gpu', 'The device used to train')

tf.app.flags.DEFINE_string(
    'optimizer', 'sgd',
    'The name of the optimizer, one of "sgd", "adam".')
tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')




FLAGS = tf.app.flags.FLAGS


BATCH_NUM = 100
FEATURE_SIZE = 100
EMBEDDED_SIZE = 3
NUM_CHANNELS = 1
CLASS_SIZE = 100
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.8

def random_shuffle(data, label):
    idx = np.random.permutation(len(data))
    random_data, random_label = data[idx], label[idx]
    return random_data, random_label

def preprocess_label(label, size):
    label_one_hot = np.zeros(size)
    label_one_hot[label] = 1
    return label_one_hot
    
def preprocess_labels(labels, size):
    labels_one_hot = []
    for label in labels:
        label_one_hot = np.zeros(size)
        label_one_hot[label] = 1
        labels_one_hot.append(label_one_hot)
    return labels_one_hot
    
def generate_input_list(ent_embedding, rel_embedding, data_id):
    hrs = []
    rts = []
    label_hrs = []
    label_rts = []
    zero_array = np.zeros(100)
    for triplet_id in data_id:
        h = ent_embedding[triplet_id[0]]
        r = rel_embedding[triplet_id[2]]
        t = ent_embedding[triplet_id[1]]
        hr = np.transpose([[h, r, zero_array]])
        rt = np.transpose([[zero_array, r, t]])
        hrs.append(hr)
        rts.append(rt)
        label_hrs.append(triplet_id[1])
        label_rts.append(triplet_id[1])
    input_list = np.concatenate((hrs, rts), axis=0)
    label_list = np.concatenate((label_hrs, label_rts), axis=0)
    return input_list, label_list

def model():
    '''Create the Model
    '''
    
    #input
    # []: dimension
    
    x = tf.placeholder(tf.float32, [FLAGS.batch_size,
                                   FEATURE_SIZE,
                                   EMBEDDED_SIZE,
                                   NUM_CHANNELS],
                          name = 'x-input')
    # x_hat = tf.placeholder(tf.float32, [BATCH_SIZE,
    #                                FEATURE_SIZE,
    #                                EMBEDDED_SIZE,
    #                                NUM_CHANNELS],
    #                       name = 'x_hat-input')

    y_gt = tf.placeholder(tf.int64, [FLAGS.batch_size, CLASS_SIZE], name = 'y-input')
    is_training = tf.placeholder(tf.bool, [])

    

    # #model
    # batch_norm_params = {
    #     'is_training': is_training,
    #     'center': True,
    #     'scale': True,
    #     'decay': 0.9997,
    #     'epsilon': 0.001,
    # }

    # # Set weight_decay for weights in Conv and DepthSepConv layers.
    weight_decay=0.00004
    # stddev=0.09
    # weights_init = tf.truncated_normal_initializer(stddev=stddev)
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)

    # with slim.arg_scope([slim.conv2d],
    #                     weights_initializer=weights_init,
    #                     activation_fn=tf.nn.relu6, 
    #                     normalizer_fn=slim.batch_norm, 
    #                     weights_regularizer=regularizer):
    #     with slim.arg_scope([slim.batch_norm], **batch_norm_params):
    with slim.arg_scope([slim.conv2d],
                       activation_fn=tf.nn.relu,
                       weights_regularizer=regularizer,
                       normalizer_fn=slim.batch_norm,
                       normalizer_params={'is_training': is_training, 'decay':0.95}):
        filter_sizes = [2, 3, 4, 5]
        with tf.variable_scope('branch-x'):
            conv1_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                # Convolution Layer
                name_scope = "conv1-%s" % filter_size
                kernel_size = [filter_size, 2]
                conv1 = slim.conv2d(x, 128, kernel_size, scope = name_scope, padding='SAME')
                print(conv1)
                conv1_outputs.append(conv1)
            conv1_outputs.append(x)
            conv1_concat = tf.concat(conv1_outputs, 3)
            print(conv1_concat)

            conv2_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                # Convolution Layer
                name_scope = "conv2-%s" % filter_size
                kernel_size = [filter_size, 2]
                conv2 = slim.conv2d(conv1_concat, 256, kernel_size, scope = name_scope, padding='VALID')
                print(conv2)
                pool = slim.max_pool2d(conv2, [FEATURE_SIZE - filter_size, 1], scope = name_scope, padding='VALID')
                print(pool)
                conv2_outputs.append(pool)
            pool = slim.max_pool2d(x, [FEATURE_SIZE, 3], scope = name_scope, padding='VALID')

            print(pool)
            conv2_outputs.append(pool)
            conv2_concat = tf.concat(conv2_outputs, 3)
            print(conv2_concat)

        with tf.variable_scope('classify'):
            conv3 = slim.conv2d(conv2_concat, CLASS_SIZE, [1, 1], scope = "conv3", padding = 'SAME')
            logits = tf.reduce_mean(conv3, axis = [1, 2])
            print(pool)
            print(logits)
        
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_gt, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    correct_prediction_5 = tf.nn.in_top_k(logits, tf.argmax(y_gt, 1), k = 5)
    Recall_5 = tf.reduce_mean(tf.cast(correct_prediction_5, tf.float32))
    correct_prediction_10 = tf.nn.in_top_k(logits, tf.argmax(y_gt, 1), k = 10)
    Recall_10 = tf.reduce_mean(tf.cast(correct_prediction_10, tf.float32))
    correct_prediction_13 = tf.nn.in_top_k(logits, tf.argmax(y_gt, 1), k = 13)
    Recall_13 = tf.reduce_mean(tf.cast(correct_prediction_13, tf.float32))

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_gt, logits=logits))
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regularization_loss = tf.add_n(regularization_losses, name='regularization_loss')
    loss = cross_entropy + regularization_loss

    step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)


    if FLAGS.optimizer == 'adam':
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, step, BATCH_NUM, 
                                           LEARNING_RATE_DECAY, staircase = True)
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=0.0001)
        learning_rate = optimizer._lr
    elif FLAGS.optimizer == 'sgd':
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, step, BATCH_NUM, 
                                                   LEARNING_RATE_DECAY, staircase = True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    train_op = slim.learning.create_train_op(loss, optimizer, global_step=step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        loss = control_flow_ops.with_dependencies([updates], loss)

    #Summary
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('accuracy/Recall_5', Recall_5)
    tf.summary.scalar('accuracy/Recall_10', Recall_10)
    tf.summary.scalar('accuracy/Recall_10', Recall_13)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('loss/cross_entropy', cross_entropy)
    tf.summary.scalar('loss/regularization_loss', regularization_loss)
    tf.summary.scalar('learning_rate', learning_rate)

    merged_summary_op = tf.summary.merge_all()
    print(merged_summary_op)

    return{'x': x,
       'y_gt': y_gt,
       'is_training': is_training,
       'train_op': train_op,
       'global_step': step,
       'accuracy': accuracy,
       'Recall_5': Recall_5,
       'Recall_10': Recall_10,
       'Recall_13': Recall_13,
       'loss': loss,
       'summary': merged_summary_op}

def main(_):
    #loading data
    ent_embedding = np.loadtxt(FLAGS.entity_feature_path, delimiter=',')
    rel_embedding = np.loadtxt(FLAGS.relation_feature_path, delimiter=',')
    train_id = np.loadtxt(FLAGS.train_path, dtype = np.int32)
    valid_id = np.loadtxt(FLAGS.valid_path, dtype = np.int32)
    test_id = np.loadtxt(FLAGS.test_path, dtype = np.int32)

    train_data, train_label = generate_input_list(ent_embedding, rel_embedding, train_id)
    valid_data, valid_label = generate_input_list(ent_embedding, rel_embedding, valid_id)
    test_data, test_label = generate_input_list(ent_embedding, rel_embedding, test_id)
    
    train_data, train_label = random_shuffle(train_data, train_label)
    valid_data, valid_label = random_shuffle(valid_data, valid_label)
    
    global FEATURE_SIZE, CLASS_SIZE, BATCH_NUM
    train_size = train_data.shape[0]
    FEATURE_SIZE = train_data.shape[1]
    CLASS_SIZE = ent_embedding.shape[0]
    BATCH_NUM = train_size/FLAGS.batch_size + 1
    valid_num = valid_data.shape[0]/FLAGS.batch_size + 1
    test_num = test_data.shape[0]/FLAGS.batch_size + 1

    #load model
    net = model()
    
    saver = tf.train.Saver(max_to_keep=100)
    #start training
    with tf.Session() as sess:
        with tf.device('/'+FLAGS.device_mode +':0'):
            #write log
            summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_path,'train'), sess.graph)
            summary_valid = tf.summary.FileWriter(os.path.join(FLAGS.log_path, 'valid'), sess.graph)
            summary_test = tf.summary.FileWriter(os.path.join(FLAGS.log_path, 'test'), sess.graph)
            #initial the variables
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            # saver.restore(sess, FLAGS.checkpoint_path)
            #Train cycle
            for epoch in range(FLAGS.epoch_size):
                for i in range(BATCH_NUM):
                    offset = i * FLAGS.batch_size
                    if train_size - offset < FLAGS.batch_size:
                        train_data, train_label = random_shuffle(train_data, train_label)
                        break
                    data_batch = train_data[offset:min((offset + FLAGS.batch_size), train_size)]
                    label_batch = preprocess_labels(train_label[offset:min((offset + FLAGS.batch_size), train_size)], ent_embedding.shape[0])
                    train_dict = {net['x']: data_batch,
                                 net['y_gt']: label_batch,
                                 net['is_training']: True}
                    step, _ = sess.run([net['global_step'], net['train_op']], feed_dict = train_dict)
                    if step % 100 == 0:
                        loss, acc, summary = sess.run([net['loss'], net['accuracy'],net['summary']], feed_dict = train_dict)
                        print('Train step = {}:    loss = {},    accuracy = {}'.format(step, loss, acc))
                        summary_writer.add_summary(summary, step)

                    if step % 500 == 0:
                        # write valid
                        data_batch = valid_data[:FLAGS.batch_size]
                        label_batch = preprocess_labels(valid_label[:FLAGS.batch_size], ent_embedding.shape[0])
                        valid_dict = {net['x']: data_batch,
                                     net['y_gt']: label_batch,
                                     net['is_training']: False}
                        loss, acc, summary = sess.run([net['loss'], net['accuracy'],net['summary']], feed_dict = valid_dict)
                        print('Valid step = {}: loss = {},    accuracy = {}'.format(step, loss, acc))
                        summary_valid.add_summary(summary, step)
                        valid_data, valid_label = random_shuffle(valid_data, valid_label)
                        

                r5_list, r10_list, r13_list = [], [], []
                for j in range(test_num):
                    offset = j * FLAGS.batch_size
                    if test_data.shape[0] - offset < FLAGS.batch_size:
                        test_data, test_label = random_shuffle(test_data, test_label)
                        break
                    data_batch = test_data[offset:min((offset + FLAGS.batch_size), test_data.shape[0])]
                    label_batch = preprocess_labels(test_label[offset:min((offset + FLAGS.batch_size), test_data.shape[0])], ent_embedding.shape[0])
                    test_dict = {net['x']: data_batch,
                                 net['y_gt']: label_batch,
                                 net['is_training']: False}
                    loss, acc, r5, r10, r13, summary = sess.run([net['loss'], net['accuracy'], net['Recall_5'], net['Recall_10'], net['Recall_13'], net['summary']], feed_dict = test_dict)
                    r5_list.append(r5)
                    r10_list.append(r10)
                    r13_list.append(r13)
                    if j == 0:
                        print('Test step = {}: loss = {},    accuracy = {}'.format(step, loss, acc))
                        summary_test.add_summary(summary, step)
                        
                avg_r5=sum(r5_list) / float(len(r5_list))
                avg_r10=sum(r10_list) / float(len(r10_list))
                avg_r13=sum(r13_list) / float(len(r13_list))
                print('Epoch {}: avg_r5 = {}, avg_r10 = {}, avg_r13={}'.format(epoch, avg_r5, avg_r10, avg_r13))
                save_path = saver.save(sess, os.path.join(FLAGS.log_path, "model_{}.ckpt".format(epoch)))
                print("Model saved in file: %s" % save_path)
                


if __name__ == '__main__':
    tf.app.run()
