做一个二分法的练习，开始采用softmax激活函数，训练结果成功。  
后改成用sigmoid函数，训练时准确率一直停留在50%, 无变化。  
根据https://www.cnblogs.com/nxf-rabbit75/p/9963208.html看到   
train_generator = train_datagen.flow_from_directory(  
    train_dir,  
    target_size = (150,150),   
    batch_size = 20,   
    class_mode = 'binary'  **因为使用了binary_crossentropy损失，所以需要用二进制标签**   
)   
修改了class_mode后，问题解决。   
特此记录     

----------------------------------------------------------------------
#将图像复制到训练、验证和测试的目录

import os,shutil

orginal_dataset_dir = 'kaggle_original_data/train'
base_dir = 'cats_and_dogs_small'
os.mkdir(base_dir)#保存新数据集的目录

train_dir = os.path.join(base_dir,'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir,'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir,'test')
os.mkdir(test_dir)

\#猫、狗的训练图像目录
train_cats_dir = os.path.join(train_dir,'cats')
os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir,'dogs')
os.mkdir(train_dogs_dir)

\#猫、狗的验证图像目录
validation_cats_dir = os.path.join(validation_dir,'cats')
os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir,'dogs')
os.mkdir(validation_dogs_dir)

\#猫、狗的测试图像目录
test_cats_dir = os.path.join(test_dir,'cats')
os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir,'dogs')
os.mkdir(test_dogs_dir)

#将前1000张猫的图像复制到train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(orginal_dataset_dir,fname)
    dst = os.path.join(train_cats_dir,fname)
    shutil.copyfile(src,dst)

#将接下来500张猫的图像复制到validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(orginal_dataset_dir,fname)
    dst = os.path.join(validation_cats_dir,fname)
    shutil.copyfile(src,dst)

#将接下来的500张猫的图像复制到test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src = os.path.join(orginal_dataset_dir,fname)
    dst = os.path.join(test_cats_dir,fname)
    shutil.copyfile(src,dst)

#将前1000张狗的图像复制到train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(orginal_dataset_dir,fname)
    dst = os.path.join(train_dogs_dir,fname)
    shutil.copyfile(src,dst)

#将接下来500张狗的图像复制到validation_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(orginal_dataset_dir,fname)
    dst = os.path.join(validation_dogs_dir,fname)
    shutil.copyfile(src,dst)

#将接下来的500张狗的图像复制到test_cats_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src = os.path.join(orginal_dataset_dir,fname)
    dst = os.path.join(test_dogs_dir,fname)
    shutil.copyfile(src,dst)
 
 

#将猫狗分类的小型卷积神经网络实例化
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
 

该问题为二分类问题，所以网咯最后一层是使用sigmoid激活的

单一单元，大小为1的Dense层。 



 

 
from keras import optimizers

model.compile(loss='binary_crossentropy',
             optimizer = optimizers.RMSprop(lr=1e-4),
             metrics = ['acc'])
 
loss: binary_crossentropy

优化器: RMSprop

度量:acc精度

 
#使用ImageDataGenerator从目录中读取图像
#ImageDataGenerator可以快速创建Python生成器，能够将硬盘上的图像文件自动转换为预处理好的张量批量
from keras.preprocessing.image import ImageDataGenerator

#将所有图像乘以1/255缩放
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary'  #因为使用了binary_crossentropy损失，所以需要用二进制标签
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary'
)
 
 

 用flow_from_directory最值得注意的是directory这个参数：
它的目录格式一定要注意是包含一个子目录下的所有图片这种格式，
driectoty路径只要写到标签路径上面的那个路径即可。 
 
for data_batch,labels_batch in train_generator:
    print('data batch shape:',data_batch.shape)
    print('labels batch shape:',labels_batch.shape)
    break
data batch shape: (20, 150, 150, 3)
labels batch shape: (20,)
#利用批量生成器拟合模型
history = model.fit_generator(
    train_generator,
    steps_per_epoch = 50,
    epochs = 30,
    validation_data = validation_generator,
    validation_steps = 50#需要从验证生成器中抽取50个批次用于评估
)
  #保存模型
  model.save('cats_and_dogs_small_1.h5')

 
  from keras.models import load_model
  model = load_model('cats_and_dogs_small_1.h5')

 手残，误操作，还好我已经保存了模型，用这句话就可以载入模型
#绘制损失曲线和精度曲线
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training_acc')
plt.plot(epochs,val_acc,'b',label='Validation_acc')
plt.title('Traing and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation_loss')
plt.title('Traing and validation loss')
plt.legend()

plt.show()
------------------------------------------------------------------------
