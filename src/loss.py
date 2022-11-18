'''loss'''
import mindspore.nn as nn
import mindspore.ops as ops


class MyLoss(nn.Cell):
    ''''Myloss'''
    def __init__(self):
        super(MyLoss, self).__init__()
        self.Norm = ops.L2Normalize(axis=1)
        self.criterion = nn.MSELoss(reduction='sum')

    def construct(self, fs_list, ft_list):
        '''construct'''
        tot_loss = 0
        for i in range(len(ft_list)):
            fs = fs_list[i]
            ft = ft_list[i]
            _, _, h, w = fs.shape
            fs_norm = self.Norm(fs)
            ft_norm = self.Norm(ft)
            f_loss = (0.5 / (w * h)) * self.criterion(fs_norm, ft_norm)
            tot_loss += f_loss

        return tot_loss
