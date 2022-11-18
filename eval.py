"""eval"""
import os
import argparse

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import ops
from src.dataset import createDataset
from src.stpm import STPM

parser = argparse.ArgumentParser(description='test')

parser.add_argument('--category', type=str, default='screw')
parser.add_argument('--device_id', type=int, default=0, help='Device id')
parser.add_argument('--data_url', type=str, default="/")
parser.add_argument('--ckpt_path', type=str, default='./')
parser.add_argument('--num_class', type=int, default=1000, help="the num of class")
parser.add_argument('--out_size', type=int, default=256, help="out size")

args = parser.parse_args()


def cal_anomaly_map(fs_list, ft_list, out_size=224):
    """cal_anomaly_map"""
    unsqueeze = ops.ExpandDims()
    Sum = ops.ReduceSum(keep_dims=False)
    Norm = ops.L2Normalize(axis=1)
    amap_mode = 'mul'
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
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
        a_map = cv2.resize(a_map, (256, 256))
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE,
                        device_target='Ascend',
                        save_graphs=False,
                        device_id=args.device_id)

    _, ds_test = createDataset(args.data_url, args.category)

    net = STPM(args, is_train=False)
    param = load_checkpoint(os.path.join(args.ckpt_path))
    load_param_into_net(net, param)
    net.set_train(False)

    gt_list_px_lvl = []
    pred_list_px_lvl = []
    gt_list_img_lvl = []
    pred_list_img_lvl = []

    for data in ds_test.create_dict_iterator():
        gt = data['gt']
        label = data['label']

        features_s, features_t = net(data['img'])
        amap = cal_anomaly_map(features_s, features_t, out_size=args.out_size)
        gt_np = gt.asnumpy()[0, 0].astype(int)

        gt_list_px_lvl.extend(gt_np.ravel())
        pred_list_px_lvl.extend(amap.ravel())
        gt_list_img_lvl.append(label.asnumpy()[0])
        pred_list_img_lvl.append(amap.max())

    pixel_auc = roc_auc_score(gt_list_px_lvl, pred_list_px_lvl)
    img_auc = roc_auc_score(gt_list_img_lvl, pred_list_img_lvl)

    print("category: ", args.category)
    print("Total pixel-level auc-roc score : ", pixel_auc)
    print("Total image-level auc-roc score :", img_auc)
