import tensorflow as tf
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.callbacks import EarlyStopping, TensorBoard
import argparse
import os

from constants import *
from dataset import *
from generate import *
from midi_util import midi_encode
from model import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    tf.config.run_functions_eagerly(True)
    models = build_or_load(allow_load=False)
    train(models)

def train(models):
    print('Loading data')
    train_data, train_labels = load_all(styles, BATCH_SIZE, SEQ_LEN)

    cbs = [
        ModelCheckpoint(MODEL_FILE, monitor='loss', save_best_only=True, save_weights_only=True),
        EarlyStopping(monitor='loss', patience=5),
        TensorBoard(log_dir='out/logs', histogram_freq=1)
    ]

    print('Compile')
    models[0].compile(run_eagerly=True)

    print('Training')
    history = models[0].fit(train_data, train_labels, epochs=4, callbacks=cbs, batch_size=BATCH_SIZE)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()
