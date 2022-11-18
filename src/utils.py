"""utils"""
import cv2
import numpy as np
from mindspore import ops


def cal_anomaly_map(fs_list, ft_list, out_size=224):
    """cal_anomaly_map"""
    unsqueeze = ops.ExpandDims()
    Sum = ops.ReduceSum(keep_dims=False)
    Norm = ops.L2Normalize(axis=1)
    anomaly_map = np.ones([out_size, out_size])
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        fs_norm = Norm(fs)
        ft_norm = Norm(ft)
        num = fs_norm * ft_norm
        cos = Sum(num, 1)
        a_map = 1 - cos
        a_map = unsqueeze(a_map, 1)
        a_map = a_map[0, 0, :, :].asnumpy()
        a_map = cv2.resize(a_map, (out_size, out_size))
        anomaly_map *= a_map
    return anomaly_map
