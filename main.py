# @Time     : Jan. 02, 2019 22:17
# @Author   : Veritas YIN
# @FileName : main.py
# @Version  : 1.0
# @Project  : Orion
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import os
#新加卷 1 line
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join as pjoin

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # 对每个 GPU 设备设置内存增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)
'''
from utils.math_graph import *
from data_loader.data_utils import *
from models.trainer import model_train
from models.tester import model_test

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_route', type=int, default=228)
parser.add_argument('--n_his', type=int, default=12)
parser.add_argument('--n_pred', type=int, default=9)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--save', type=int, default=10)
parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--inf_mode', type=str, default='merge')

args = parser.parse_args()
print(f'Training configs: {args}')

n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt
# blocks: settings of channel size in st_conv_blocks / bottleneck design
blocks = [[1, 32, 64], [64, 32, 128]]

# Load wighted adjacency matrix W
if args.graph == 'default':
    W = weight_matrix(pjoin('./dataset', f'PeMSD7_W_{n}.csv'))
else:
    # load customized graph weight matrix
    W = weight_matrix(pjoin('./dataset', args.graph))

# Calculate graph kernel
L = scaled_laplacian(W)
# Alternative approximation method: 1st approx - first_approx(W, n).
Lk = cheb_poly_approx(L, Ks, n)


#***
#tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))
graph_kernel_variable = tf.Variable(tf.cast(Lk, tf.float32), name='graph_kernel')

# 创建一个包含此变量的模型类（如果你需要的话）
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.graph_kernel = graph_kernel_variable

    def call(self, inputs):
        # 使用 graph_kernel 在模型中进行操作
        pass
model = MyModel()

# 如果需要保存变量，可以使用 TensorFlow 的保存机制
tf.saved_model.save(model, 'path_to_save_model')






# Data Preprocessing
data_file = f'PeMSD7_V_{n}.csv'
n_train, n_val, n_test = 34, 5, 5
PeMS = data_gen(pjoin('./dataset', data_file), (n_train, n_val, n_test), n, n_his + n_pred)
print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')

if __name__ == '__main__':
    model_train(PeMS, blocks, args)
    model_test(PeMS, PeMS.get_len('test'), n_his, n_pred, args.inf_mode)
