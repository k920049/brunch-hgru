import tensorflow as tf
from tensorflow.python.keras.layers.cudnn_recurrent import *

cudnnGRUCell = CuDNNGRU(units = 200).cell
print(cudnnGRUCell)