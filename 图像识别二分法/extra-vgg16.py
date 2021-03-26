from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import Xception
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import tensorflow as tf
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.applications.vgg16 import decode_predictions
import os
from random import shuffle
from tensorflow import keras
from keras import layers
import random
from keras.preprocessing.image import ImageDataGenerator


def loadimg(file):
    image = load_img(file, target_size=(299, 299))
    # image = img_to_array(image)
    image = img_to_array(image) / 255.0
    # image = np.expand_dims(image, axis=0)
    # print(image.shape)
    return image


def load_data():
    files = list_files()
    X = []
    Y = []
    for file in files:
        X.append(loadimg(file[1]))
        if file[1].find('female'):
            Y.append([0, 1.0])
        else:
            Y.append([1.0, 0])
    return X, Y


def load_customer_data():
    (x_train, y_train), (x_test, y_test) =   tf.keras.datasets.mnist.load_data()
    print(x_test.shape)
    print(y_train.shape)
    return x_train, y_train


def predict_vgg():
    # model = VGG16(weights="imagenet")
    model = Xception(weights='imagenet')

    image = []
    image.append(loadimg('/home/ai/temp/train_data/girl/girl.401.jpg'))
    image = np.array(image)

    y_pred = model.predict(image)
    # print(y_pred)
    label = decode_predictions(y_pred)
    print(label)
   # # 检索最可能的结果，例如最高的概率
    label = label[0][0]

    # 打印结果
    print("%s (%.2f%%)" % (label[1],label[2]*100))


def list_files():
    file_list = []
    # for root, dirs, files in os.walk(r"/home/ai/temp/train_data"):
    for root, dirs, files in os.walk(r"/home/ai/temp/Training"):
        for file in files:
            # 获取文件所属目录
            # print(file)
            file_list.append((file, os.path.join(root, file)))
            # 获取文件路径
            # print(os.path.join(root, file))
    # shuffle(file_list)
    return file_list


def generator():
    train_data = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_data.flow_from_directory('/home/ai/temp/train_data', target_size=(299, 299), color_mode='rgb', class_mode='binary',
                                   batch_size=50, shuffle=True)
    print(train_generator.class_indices)
    # for data_batch, labels_batch in train_generator:
    #     print('data batch shape:', data_batch.shape)
    #     print('labels batch shape:', labels_batch.shape)
    #     break

    return train_generator

    # files = list_files()
    # batch_size = 20
    # while 1:
    #     shuffle(files)
    #     x = []
    #     y = []
    #     sample = random.sample(files, batch_size)
    #     for file in sample:
    #         img = loadimg(file[1])
    #         x.append(img)
    #         if file[1].find('female')>=0:
    #             y.append([0, 1.0])
    #             # y.append(0)
    #         else:
    #             y.append([1.0, 0])
    #             # y.append(1)
    #     x = np.array(x)
    #     y = np.array(y)
    #     # print(x.shape, y.shape)
    #     yield x, y

    # x_train, y_train = load_customer_data()
    # while 1:
    #     batch = np.random.randint(0, len(x_train), size=5)
    #     x = x_train[0:200]
    #     y = y_train[0:200]
    #     yield x, y


def train():
    model_xception = Xception(weights="imagenet", include_top=False, input_shape=(299,299,3))

    model_xception.trainable = False

    # model_vgg16 = VGG16(weights="imagenet")
    # model_vgg19 = VGG19(weights="imagenet")
    # model_resnet50 = ResNet50(weights = "imagenet")
    # model_inceptionv3 = InceptionV3(weights="imagenet")

    model = tf.keras.models.Sequential()
    model.add(model_xception)
    model.add(Flatten(input_shape=model.output_shape[1:]))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    # model.add(Dense(2, activation='softmax'))
    model.add(Dense(1, activation='sigmoid'))
    # model.compile(optimizer="sgd", loss='categorical_crossentropy',metrics=['acc'])
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    model.fit_generator(generator(), epochs=5, steps_per_epoch=20, verbose=1)

    model.save('/home/ai/temp/Xception.sigmoid.h5')


def create_customer_model():

    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Flatten(input_shape=(299, 299, 3)),
    #     # tf.keras.layers.Flatten(input_shape=(28, 28)),
    #     tf.keras.layers.Dense(512, activation='relu'),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Dense(256, activation='relu'),
    #     tf.keras.layers.Dropout(0.5),
    #     tf.keras.layers.Dense(2, activation='softmax')
    # ])
    #
    # model.compile(optimizer='adam',
    #               loss='sparse_categorical_crossentropy', metrics=['acc'])
    model = keras.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(299, 299, 3)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(200, 200, 3)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(200, 200, 3)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(236, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['acc'])
    model.summary()
    return model


def procee_resut(value):
    # if value[0][0]>value[0][1]:
    #     return 'boy'
    # else:
    #     return 'girl'
    if value>0.5:
        return 'girl'
    else:
        return 'boy'


def predict(label, index):
    # '/home/ai/temp/train_data/girl/girl.438.jpg'
    root = r'/home/ai/temp/train_data/'
    file = '%s%s/%s.%d.jpg' % (root, label, label, index)
    result = predict_vgg_ex(file)
    print( result, label, str(index))


def predict_vgg_ex(file):
    img = []
    img.append(loadimg(file))
    # img.append(loadimg('/home/ai/temp/Training/male/097171.jpg.jpg'))
    # img = preprocess_input(img)
    img = np.array(img)

    model = tf.keras.models.load_model('/home/ai/temp/Xception.sigmoid.h5')
    # label = model.predict(img)
    print(model.predict(img))
    # print(label)
    # return procee_resut(label)
    label = model.predict_classes(img)
    if label == 0:
        return 'boy'
    else:
        return 'girl'


def show_model():
    model = tf.keras.models.Sequential()
    model.load_weights(r'/home/ai/temp/speech_model251_e_0_step_625000.model')


if __name__ == '__main__':
    # show_model()
    # run_vgg()
    train()

    # p_data = [('boy', 345),('girl',2001),('boy',2001),('girl',2003),('girl',2004)]
    # p_data = [('girl',1139)]
    # for d in p_data:
    #     predict(d[0], d[1])
