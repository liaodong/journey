import tensorflow as tf
from transformers import BertTokenizer
from tensorflow import keras
from tqdm import tqdm
import numpy as np
import pandas as pd
from multiprocessing.pool import Pool

file_path = 'data/data_train.txt'
model_path = "bert-base-chinese"



def create_model():
    vocab_size = 38000
    model = keras.Sequential()
    '''
      对文本数据进行平均池化操作
      输入得数据格式为：(batch_size, steps, features)
      batch_size表示本批次有多少条文本
      steps表示一个文本里面有是多少个单词
      features表示一个单词使用多少维度进行表示
      输出得数据格式为：(batch_size, features)
      features表示一行文本使用多少个维度进行表示
      所以从上面得输入和输出可以看出。steps这个维度不见了，
      所以这允许模型以尽可能最简单的方式处理可变长度的输入
      原文链接：https://blog.csdn.net/weixin_43824178/article/details/99182766
    '''
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(32, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    return model


def preprocess():
    data = []
    label = []
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # df = pd.read_csv(file_path, sep='\t', header=None)
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='Processing'):
            t = line.split('\t')
            data.append(tokenizer.encode(t[1], add_special_tokens=False, ))
            label.append(int(t[0]))
    print('data size = ', len(data))

    return data, label
    pass


def train(train_data, train_label):
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=0,
                                                            padding='post',
                                                            maxlen=64)
    rate = 0.8
    length = len(train_data)
    train_x = train_data[: int(length * rate)]
    train_y = train_label[:int(length * rate)]

    val_x = train_data[-(int(length*(1-rate))):]
    val_y = train_label[-(int(length*(1-rate))):]

    model = create_model()
    model.fit(train_x, train_y, epochs=5, batch_size=100, validation_data=(val_x, val_y))
    model.save('model/mysms2', save_format='tf')

    pass

def my_encode(input):
    #通过自定义方法，解决传提多个参数到Pool().map中
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return tokenizer.encode(input, add_special_tokens=False)

def predict(inputs):

    encoded = Pool().map(my_encode, inputs)
    x = np.array(encoded)
    x = keras.preprocessing.sequence.pad_sequences(x,
                                                value=0,
                                                padding='post',
                                                maxlen=64)
    model = tf.keras.models.load_model('model/model.h5')
    model.summary()
    outputs = model.predict(x)
    print(outputs)
    pass


if __name__ == '__main__':
    # train_data, train_label = preprocess(0)
    # train(train_data, train_label)
    predict(['您值得拥有',
             '今天最后三天降价'])
