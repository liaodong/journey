---model.fit_generator()函数参数
·
fit_generator(self, generator, steps_per_epoch, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_q_size=10, workers=1, pickle_safe=False, initial_epoch=0)
·
利用Python的生成器，逐个生成数据的batch并进行训练。生成器与模型将并行执行以提高效率。例如，该函数允许我们在CPU上进行实时的数据提升，同时在GPU上进行模型训练
函数的参数是：
generator：生成器函数，生成器的输出应该为：
一个形如（inputs，targets）的tuple
一个形如（inputs, targets,sample_weight）的tuple。所有的返回值都应该包含相同数目的样本。生成器将无限在数据集上循环。每个epoch以经过模型的样本数达到samples_per_epoch时，记一个epoch结束
steps_per_epoch：整数，当生成器返回steps_per_epoch次数据时计一个epoch结束，执行下一个epoch
epochs：整数，数据迭代的轮数
verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
validation_data：具有以下三种形式之一
生成验证集的生成器
一个形如（inputs,targets）的tuple
一个形如（inputs,targets，sample_weights）的tuple
validation_steps: 当validation_data为生成器时，本参数指定验证集的生成器返回次数
class_weight：规定类别权重的字典，将类别映射为权重，常用于处理样本不均衡问题。
sample_weight：权值的numpy array，用于在训练时调整损失函数（仅用于训练）。可以传递一个1D的与样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，传递一个的形式为（samples，sequence_length）的矩阵来为每个时间步上的样本赋不同的权。这种情况下请确定在编译模型时添加了sample_weight_mode='temporal'。
workers：最大进程数
max_q_size：生成器队列的最大容量
pickle_safe: 若为真，则使用基于进程的线程。由于该实现依赖多进程，不能传递non picklable（无法被pickle序列化）的参数到生成器中，因为无法轻易将它们传入子进程中。
initial_epoch: 从该参数指定的epoch开始训练，在继续之前的训练时有用。
函数返回一个History对象
例子：
·
def generate_arrays_from_file(path):
    while 1:
            f = open(path)
            for line in f:
                # create Numpy arrays of input data
                # and labels, from each line in the file
                x, y = process_line(line)
                yield (x, y)
        f.close()
model.fit_generator(generate_arrays_from_file('/my_file.txt'),
        samples_per_epoch=10000, epochs=10)
·
