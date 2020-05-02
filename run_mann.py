from mann.model import MANN
from mann.utils.generators import OmniglotGenerator

from argparse import ArgumentParser
import tensorflow as tf
import os
import numpy as np
import datetime


def build_argparser():
    parser = ArgumentParser()

    parser.add_argument('--mode', default="train")
    parser.add_argument('--restore_training', default=False)
    parser.add_argument('--batch-size',
            dest='_batch_size',	help='Batch size (default: %(default)s)',
            type=int, default=16)
    parser.add_argument('--num-classes',
            dest='_nb_classes', help='Number of classes in each episode (default: %(default)s)',
            type=int, default=5)
    parser.add_argument('--num-samples',
            dest='_nb_samples_per_class', help='Number of taotal samples in each episode (default: %(default)s)',
            type=int, default=10)
    parser.add_argument('--input-height',
            dest='_input_height', help='Input image height (default: %(default)s)',
            type=int, default=20)
    parser.add_argument('--input-width',
            dest='_input_width', help='Input image width (default: %(default)s)',
            type=int, default=20)
    parser.add_argument('--num-reads',
            dest='_nb_reads', help='Number of read heads (default: %(default)s)',
            type=int, default=4)
    parser.add_argument('--controller-size',
            dest='_controller_size', help='Number of hidden units in controller (default: %(default)s)',
            type=int, default=200)
    parser.add_argument('--memory-locations',
            dest='_memory_locations', help='Number of locations in the memory (default: %(default)s)',
            type=int, default=128)
    parser.add_argument('--memory-word-size',
            dest='_memory_word_size', help='Size of each word in memory (default: %(default)s)',
            type=int, default=40)
    parser.add_argument('--num_layers',
                        dest='_num_layers', help='Size of each word in memory (default: %(default)s)',
                        type=int, default=1)
    parser.add_argument('--learning-rate',
            dest='_learning_rate', help='Learning Rate (default: %(default)s)',
            type=float, default=1e-3)
    parser.add_argument('--start_iterations',
            dest='_start_iterations', default=0)
    parser.add_argument('--iterations',
            dest='_iterations', help='Number of iterations for training (default: %(default)s)',
            type=int, default=100000)
    parser.add_argument('--augment', default=True)
    parser.add_argument('--save-dir', default='./ckpt/')
    parser.add_argument("--log-dir", default="./log/")
    parser.add_argument('--model', default="MANN", help='LSTM or MANN')

    return parser

def metric_accuracy(args, labels, outputs):
    seq_length = args._nb_classes * args._nb_samples_per_class
    outputs = np.argmax(outputs, axis=-1)
    correct = [0] * seq_length
    total = [0] * seq_length
    for i in range(np.shape(labels)[0]):
        label = labels[i]
        output = outputs[i]
        class_count = {}
        for j in range(seq_length):
            class_count[label[j]] = class_count.get(label[j], 0) + 1
            total[class_count[label[j]]] += 1
            if label[j] == output[j]:
                correct[class_count[label[j]]] += 1
    return [float(correct[i]) / total[i] if total[i] > 0. else 0. for i in range(1, args._nb_samples_per_class + 1)]


def train(model:MANN, data_genarator: OmniglotGenerator, sess, saver, args):
    start_iter = args._start_iterations
    max_iter = args._iterations
    csv_write_path = '{}/{}-{}-{}--{}.csv'.format(args.log_dir, args.model, args._nb_classes, args._nb_samples_per_class, datetime.datetime.now().strftime('%m-%d-%H-%M'))

    print(args)
    print("1st\t2nd\t3rd\t4th\t5th\t6th\t7th\t8th\t9th\t10th\tbatch\tloss")

    for ep in range(start_iter, max_iter):
        if ep % 100 == 0:
            image, label = data_genarator.sample("test", args._batch_size)
            feed_dict = {model.image: image, model.label: label}
            output, loss = sess.run([model.output, model.loss], feed_dict=feed_dict)
            accuracy = metric_accuracy(args, label, output)
            for accu in accuracy:
                print('%.4f' % accu, end='\t')
            print('%d\t%.4f' % (ep, loss))

            with open(csv_write_path, 'a') as fh:
                fh.write(str(ep) + ", " +", ".join(['%.4f' % accu for accu in accuracy])+ "\n")


        if ep % 5000 == 0 and ep > 0:
            saver.save(sess, os.path.join(args.save_dir, args.model) + "/model.", global_step=ep)

        image, label = data_genarator.sample("train", args._batch_size)
        feed_dict = {model.image: image, model.label: label}

        sess.run(model.train_op, feed_dict=feed_dict)

def test(model: MANN, data_generator: OmniglotGenerator, sess, args):
    print("Test Result\n1st\t2nd\t3rd\t4th\t5th\t6th\t7th\t8th\t9th\t10th\tloss")
    label_list = []
    output_list = []
    loss_list = []
    for ep in range(100):
        image, label = data_generator.sample("test", args._batch_size)
        feed_dict = {model.image: image, model.label: label}
        output, loss = sess.run([model.output, model.loss], feed_dict = feed_dict)
        label_list.append(label)
        output_list.append(output)
        loss_list.append(loss)
    accuracy = metric_accuracy(args, np.concatenate(label_list, axis=0), np.concatenate(output_list, axis=0))
    for accu in accuracy:
        print('%.4f' % accu, end='\t')
    print(np.mean(loss_list))

if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()

    batch_size = args._batch_size
    nb_classes = args._nb_classes
    nb_samples_per_class = args._nb_samples_per_class
    img_size = (args._input_height, args._input_width)
    input_size = args._input_height * args._input_width

    nb_reads = args._nb_reads
    controller_size = args._controller_size
    memory_size = args._memory_locations
    memory_dim = args._memory_word_size
    num_layers = args._num_layers

    learning_rate = args._learning_rate



    model = MANN(learning_rate, input_size, memory_size, memory_dim,
                 controller_size, nb_reads, num_layers, nb_classes, nb_samples_per_class, batch_size, args.model)
    model.build_model()

    data_generator = OmniglotGenerator(data_folder="./omniglot", nb_classes=nb_classes,
                                       nb_samples_per_class=nb_samples_per_class, img_size=img_size)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=tf_config)

    if args.restore_training or args.mode == "test":
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(os.path.join(args.save_dir, args.model))
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())

    if args.mode == "train":
        train(model, data_generator, sess, saver, args)
    elif args.mode == "test":
        test(model, data_generator, sess, args)

    sess.close()
