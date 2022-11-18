"""
python pth2ckpt.py
"""
import argparse
import torch
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor

param = {
    'bn1.bias': 'bn1.beta',
    'bn1.weight': 'bn1.gamma',
    'IN.weight': 'IN.gamma',
    'IN.bias': 'IN.beta',
    'BN.bias': 'BN.beta',
    'in.weight': 'in.gamma',
    'bn.weight': 'bn.gamma',
    'bn.bias': 'bn.beta',
    'bn2.weight': 'bn2.gamma',
    'bn2.bias': 'bn2.beta',
    'bn3.bias': 'bn3.beta',
    'bn3.weight': 'bn3.gamma',
    'BN.running_mean': 'BN.moving_mean',
    'BN.running_var': 'BN.moving_variance',
    'bn.running_mean': 'bn.moving_mean',
    'bn.running_var': 'bn.moving_variance',
    'bn1.running_mean': 'bn1.moving_mean',
    'bn1.running_var': 'bn1.moving_variance',
    'bn2.running_mean': 'bn2.moving_mean',
    'bn2.running_var': 'bn2.moving_variance',
    'bn3.running_mean': 'bn3.moving_mean',
    'bn3.running_var': 'bn3.moving_variance',
    'downsample.1.running_mean': 'down_sample_layer.1.moving_mean',
    'downsample.1.running_var': 'down_sample_layer.1.moving_variance',
    'downsample.0.weight': 'down_sample_layer.0.weight',
    'downsample.1.bias': 'down_sample_layer.1.beta',
    'downsample.1.weight': 'down_sample_layer.1.gamma'
}


def pytorch2mindspore(pt_path):
    """

    Returns:
        object:
    """
    par_dict = torch.load(pt_path, map_location='cpu')

    new_params_list = []

    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]
        print("key name: ", name)
        for fix in param:
            if name.endswith(fix):
                name = name[:name.rfind(fix)]
                name = name + param[fix]

        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.detach().numpy())
        new_params_list.append(param_dict)

    save_checkpoint(new_params_list, 'STPM-99.ckpt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--pt_path', type=str, help="/path/resnet18-f37072fd.pth")
    args = parser.parse_args()
    pytorch2mindspore(args.pt_path)
