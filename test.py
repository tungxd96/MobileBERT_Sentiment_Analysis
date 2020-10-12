import tensorflow as tf
import torch
import numpy as np

# t1 = tf.constant([[0, 1, 2, 3]])
# t2 = tf.constant([[2, 3, 4, 5]])
# # print(t1, t2)
# t1 = tf.concat([t1, t2], axis=0)
# for i in range(len(t1), 10):
#     t = tf.constant([[0]])
#     t2 = tf.concat([t2, t], axis=-1)
# # print(t2)

a = [[2, 131, 806, 560, 1, 3], [2, 1, 790, 1389, 666, 439, 58, 1, 35, 124, 16, 966, 78, 54, 81, 803, 53, 44, 825, 108, 1025, 80, 1442, 364, 30, 1895, 48, 31, 449, 111, 324, 49, 108, 21, 1491, 107, 1150, 1824, 149, 1, 966, 78, 54, 81, 1, 3]]
b = [[0, 1, 2], [3, 4, 5]]
# c = torch.tensor(np.array(a))
d = tf.constant(b)
print(d.numpy())