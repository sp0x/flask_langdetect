from cnndetector import train, predict
import tensorflow as tf
from pathlib import Path
import os
import pickle
import numpy as np

from cnn import Model, CnnDetectorContext

__config = dict()
__model = None
__context = None


def unpickle(path):
    with open(path, 'rb') as f:
        out = pickle.load(f)
    return out


def loadConfig():
    global __model
    global __config
    global __context
    if __model is not None:
        return __model
    current_dir = os.path.abspath(os.path.dirname(__file__))
    train_dir = os.path.join(current_dir, 'models', '1543501620')
    data_dir = os.path.join(current_dir, 'data', 'ted500')

    config = unpickle(os.path.join(train_dir, 'flags.cPickle'))
    __config = config
    config['train_dir'] = train_dir
    config['data_dir'] = data_dir
    if 'num_classes' not in config:
        config['num_classes'] = 65
    if 'sent_len' not in config:
        config['sent_len'] = 257
    __context = CnnDetectorContext(config)


def predict_batch(x, raw_text=True):
    loadConfig()
    vocab = __context.vocab
    class_names = vocab.class_names
    train_dir = __config['train_dir']
    ckpt = tf.train.get_checkpoint_state(train_dir)
    # Since this might be on a different FS mmodify the path
    # Todo: check for this
    model_filename = Path(ckpt.model_checkpoint_path).name
    model_file = os.path.join(train_dir, model_filename)
    __config['model_checkpoint_path'] = model_file
    __config['model_filename'] = model_filename
    with tf.Graph().as_default():
        with tf.variable_scope('cnn'):
            with tf.Session() as sess:
                if not (ckpt and ckpt.model_checkpoint_path):
                    raise IOError("Loading checkpoint file failed!")
                m = Model(__config, is_train=False)
                saver = tf.train.Saver(tf.global_variables(), allow_empty=True)
                saver.restore(sess, model_file)
                for i, xi in enumerate(x):
                    try:
                        # Convert the text to a vector of ids
                        if raw_text:
                            tid = vocab.text2id(xi)
                            if tid is None:
                                yield ['inv', []]
                            xi = tid
                            x_input = np.array([xi])
                        else:
                            x_input = xi
                        # Predict
                        scores = sess.run(m.scores, feed_dict={m.inputs: x_input})

                        if raw_text:
                            best_score = int(np.argmax(scores))
                            # scores = [float(str(i)) for i in scores[0]]
                            y_pred = class_names[best_score]
                            # scores = dict(zip(class_names, scores))
                        else:
                            y_pred = np.argmax(scores, axis=1)
                        yield y_pred
                    except Exception as e:
                        yield ['inv', []]



if __name__ == "__main__":
    values = ["This an example of something to get the language from"]
    result = list(predict_batch(values))
    for i, test in enumerate(values):
        language = result[i]
        print("{0} : {1}".format(language, test))
