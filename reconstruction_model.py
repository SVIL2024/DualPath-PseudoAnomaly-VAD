import math
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


class inconv(nn.Module):
    '''
    inconv only changes the number of channels
    '''

    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1),
            double_conv(out_ch, out_ch),
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, op="none"): # 64 - 32
        super(up, self).__init__()
        self.op = op

        self.up = nn.ConvTranspose3d(in_channels=in_ch, out_channels=in_ch // 2, kernel_size=3, stride=2, padding=1,
                                     output_padding=1)  # 逆卷积， 上采样
        assert op in ["concat", "none"]

        if op == "concat":
            self.conv = double_conv(in_ch, out_ch)
        else:
            self.conv = double_conv(out_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)

        if self.op == "concat":
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1

        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0., epsilon=1e-12):
    output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)  # 大于0，硬收缩
    return output

class Memory(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025, choices=None):
        super(Memory, self).__init__()
        self.mem_dim = mem_dim  # 插槽
        self.fea_dim = fea_dim
        self.choice = choices

        # if self.choice is not None:
        #     self.memMatrix = nn.Parameter(torch.empty(mem_dim, fea_dim)).cpu()
        # else:
        #     self.memMatrix = nn.Parameter(torch.empty(mem_dim, fea_dim))  # M,C
        self.memMatrix = nn.Parameter(torch.empty(mem_dim, fea_dim))
        self.shrink_thres = shrink_thres

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.memMatrix.size(1))
        self.memMatrix.data.uniform_(-stdv, stdv)  # 随机化参数

    def forward(self, x):
        """
        :param x: query features with size [N,C], where N is the number of query items,
                  C is same as dimension of memory slot

        :return: query output retrieved from memory, with the same size as x.
        """
        # dot product
        # if self.choice is not None:
        #     att_weight = F.linear(input=x, weight=self.memMatrix).cpu()  # [N,C] by [M,C]^T --> [N,M] 线性变换
        # else:
        #     att_weight = F.linear(input=x, weight=self.memMatrix)
        att_weight = F.linear(input=x, weight=self.memMatrix)  # [N,C] by [M,C]^T --> [N,M] 线性变换
        att_weight = F.softmax(att_weight, dim=1)  # NxM

        # if use hard shrinkage
        if self.shrink_thres > 0:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)  # [N,M]
            # normalize
            att_weight = F.normalize(att_weight, p=1, dim=1)  # [N,M]

        # out slot
        out = F.linear(att_weight, self.memMatrix.permute(1, 0))  # [N,M] by [M,C]  --> [N,C] N是查询项的数目，C与记忆槽的维数相同

        return dict(out=out, att_weight=att_weight, mem=self.memMatrix)

class Mem(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(Mem, self).__init__()
        self.memory = Memory(mem_dim=mem_dim, fea_dim=fea_dim, shrink_thres=shrink_thres)
    def forward(self, x3):
        bs, C, D, H, W = x3.shape  # 解析维度
        x_flat = x3.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        mem_result = self.memory(x_flat)
        x_flat_out = mem_result["out"]
        att_weight = mem_result["att_weight"]
        x3_out = x_flat_out.view(bs, D, H, W, C).permute(0, 4, 1, 2, 3)
        out = dict(x3_out=x3_out, att_weight=att_weight)
        return out

class Encoder(nn.Module):
    def __init__(self, num_in_ch, features_root):
        super(Encoder, self).__init__()
        self.in_conv = inconv(num_in_ch, features_root)
        self.down_1 = down(features_root, features_root * 2)
        self.down_2 = down(features_root * 2, features_root * 4)
        self.down_3 = down(features_root * 4, features_root * 8)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        return x3, x2, x1, x0


class Decoder(nn.Module):
    def __init__(self, features_root, num_out_ch, skip_ops):
        super(Decoder, self).__init__()
        self.skip_ops = skip_ops
        self.up_3 = up(features_root * 8, features_root * 4, op=self.skip_ops[-1])
        self.up_2 = up(features_root * 4, features_root * 2, op=self.skip_ops[-2])
        self.up_1 = up(features_root * 2, features_root, op=self.skip_ops[-3])
        self.out_conv = outconv(features_root, num_out_ch)

    def forward(self, x3, x2, x1, x0):
        recon1 = self.up_3(x3, x2 if self.skip_ops[-1] != "none" else None)
        recon2 = self.up_2(recon1, x1 if self.skip_ops[-2] != "none" else None)
        recon3 = self.up_1(recon2, x0 if self.skip_ops[-3] != "none" else None)
        recon = self.out_conv(recon3)  # Conv3d
        return recon, recon3, recon2, recon1

class Decoder0(nn.Module):
    def __init__(self, features_root, num_out_ch, skip_ops):
        super(Decoder0, self).__init__()
        self.skip_ops = skip_ops
        self.up_3 = up(features_root * 8, features_root * 4, op=self.skip_ops[-1])
        self.up_2 = up(features_root * 4, features_root * 2, op=self.skip_ops[-2])
        self.up_1 = up(features_root * 2, features_root, op=self.skip_ops[-3])
        self.out_conv = outconv(features_root, num_out_ch)

    def forward(self, x3, x2, x1, x0):
        recon1 = self.up_3(x3, x2 if self.skip_ops[-1] != "none" else None)
        recon2 = self.up_2(recon1, x1 if self.skip_ops[-2] != "none" else None)
        recon3 = self.up_1(recon2, x0 if self.skip_ops[-3] != "none" else None)
        recon = self.out_conv(recon3)  # Conv3d
        return recon, recon3, recon2, recon1