import tensorflow as tf
from tensorflow import keras

import numpy as np
import pathlib
import codecs
import jieba


def list_str2int(data):
    for i,v in enumerate(data):
        data[i] = int(v)
    return data


def save_token_to_int(data, file_path='./data/tags_token_results_int'):
    dict = {}
    i = 1
    for x in data:
        for y in x:
            if y not in dict:
                dict[y] = i
                i += 1
    print(len(dict))
    with codecs.open(file_path + "_dict", 'w') as f:
        f.write(str(dict))
    with codecs.open(file_path, 'w', 'utf-8') as f:
        for x in data:
            line = []
            for y in x:
                v = dict.get(y)
                line.append(str(v))
            f.write(" ".join(line) + '\n')


def read_train_data():
    file_path = './data/data_train.txt'
    target = []
    data = []
    with codecs.open(file_path, 'r', 'utf-8') as f:
        for line in f.read().split('\n')[:-1]:
            line = line.strip()
            target.append(line[0])
            data.append(line[1:].lstrip())
    return data, target


def save_tokenlization_result(data, target, file_path='./data/tags_token_results'):
    with codecs.open(file_path, 'w', 'utf-8') as f:
        for x in data:
            f.write(' '.join(x) + '\n')

    with open(file_path + '_tag', 'w') as f:
        for x in target:
            f.write(x + '\n')

'''
数据预处理
1、通过jieba分词， 生成自定义分词token字典表  _dict
2、将训练短信tokenizer化       _int
3、标签化       _tag
'''
def preprocess():

    data, target = read_train_data()
    dd = []
    for d in data:
        dd.append(jieba.lcut(d))
    data = dd
    save_tokenlization_result(data, target)

    data_root = pathlib.Path("./data")
    train_data = (data_root / "tags_token_results").open(encoding='utf-8').readlines()
    train_data = [line[:-1].split(' ') for line in train_data]
    save_token_to_int(train_data)


def train():

    data_root = pathlib.Path("./data")

    train_data = (data_root / "tags_token_results_int").open(encoding="utf-8").readlines()
    train_data = [line[:-1].split(' ') for line in train_data]
    train_labels = (data_root / "tags_token_results_tag").open(encoding="utf-8").readlines()
    train_labels = list_str2int(train_labels)

    print("Training entries:{}, labels: {}".format(len(train_data), len(train_labels)))
    print(train_data[0], train_labels[0], type(train_labels[0]))
    print(len(train_data[0]), len(train_data[1]))

    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=0,
                                                            padding='post',
                                                            maxlen=64)
    print(len(train_data[0]), len(train_data[1]))
    print(train_data[0])

    # checkpoint_path = "training_1/cp-{epoch:04d}.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    # cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #     checkpoint_path, verbose=1, save_weights_only=True,
    #     period=1)

    vocab_size = 380000
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

    model.summary()

    x_val = train_data[200000:250000]
    partial_x_train = train_data[:200000]
    y_val = train_labels[200000:250000]
    partial_y_train = train_labels[:200000]

    x_val = np.array(x_val)
    partial_x_train = np.array(partial_x_train)
    y_val = np.array(y_val)
    partial_y_train = np.array(partial_y_train)

    print(partial_x_train.shape)
    print(len(partial_x_train), len(partial_y_train))

    model.fit(partial_x_train,
                        partial_y_train,
                        epochs=10,
                        #callbacks=[cp_callback],
                        batch_size=128,
                        validation_data=(x_val, y_val),
                        verbose=1)

    model.save('model/aisms',save_format='tf')


def load_dict(file_path='./data/tags_token_results_int_dict'):
    dicts = {}
    with open(file_path, 'r') as f:
        dicts = eval(f.read())
    return dicts


def predict():
    model = tf.keras.models.load_model('model/aisms', compile=True)
    model.summary()
    dict = load_dict()
    x = '广东中旅旅行社提供国内旅游及国外旅游咨询、代订酒店、机票等服务。热线电话：2323121'
    x = jieba.lcut(x)
    line = []
    for y in x:
        v = dict.get(y)
        if v is None:
            continue
        line.append(str(v))
    x = [line]
    x = keras.preprocessing.sequence.pad_sequences(x, value=0,
                                                   padding='post',
                                                   maxlen=64)
    x = np.array(x)
    print(x.shape)

    outputs = model.predict(x)
    print(outputs)
    pass


if __name__ == '__main__':
    preprocess()
    train()
    predict()

