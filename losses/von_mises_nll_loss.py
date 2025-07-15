# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import torch
import torch.nn as nn



def _eval_poly(y, coef):
    coef = list(coef)               # 多项式的系数列表，从常数项开始，到最高次项结束
    result = coef.pop()
    while coef:
        result = coef.pop() + y * result
    return result

# 函数的工作原理是计算多项式的值。它首先将 coef 转换为列表（如果它不是列表的话），然后从列表中弹出最后一个元素（常数项），并将其赋值给 result。
# 接下来，它进入一个循环，每次从系数列表中弹出一个系数，并将其与 y 乘以当前的 result 相加，直到系数列表为空。最后，返回计算出的多项式值。

# 两组系数，分别用于计算修正贝塞尔函数的 I0 和 I1 的近似值
_I0_COEF_SMALL = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.360768e-1, 0.45813e-2]
_I0_COEF_LARGE = [0.39894228, 0.1328592e-1, 0.225319e-2, -0.157565e-2, 0.916281e-2,
                  -0.2057706e-1, 0.2635537e-1, -0.1647633e-1, 0.392377e-2]
_I1_COEF_SMALL = [0.5, 0.87890594, 0.51498869, 0.15084934, 0.2658733e-1, 0.301532e-2, 0.32411e-3]
_I1_COEF_LARGE = [0.39894228, -0.3988024e-1, -0.362018e-2, 0.163801e-2, -0.1031555e-1,
                  0.2282967e-1, -0.2895312e-1, 0.1787654e-1, -0.420059e-2]

_COEF_SMALL = [_I0_COEF_SMALL, _I1_COEF_SMALL]
_COEF_LARGE = [_I0_COEF_LARGE, _I1_COEF_LARGE]


def _log_modified_bessel_fn(x, order=0):
    assert order == 0 or order == 1          # order：函数的阶数，可以是 0（对应I0函数）或 1（对应I1 函数），assert 语句确保 order 参数只能是 0 或 1



    # compute small solution
    y = (x / 3.75)
    y = y * y
    small = _eval_poly(y, _COEF_SMALL[order])
    if order == 1:
        small = x.abs() * small
    small = small.log()

    # compute large solution
    y = 3.75 / x
    large = x - 0.5 * x.log() + _eval_poly(y, _COEF_LARGE[order]).log()

    result = torch.where(x < 3.75, small, large)      #  torch.where 函数根据 x 的值选择适当的解。如果 x 小于 3.75，选择小值解；否则，选择大值解
    return result



# Von Mises 分布是一种用于表示角度数据的连续概率分布
class VonMisesNLLLoss(nn.Module):

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(VonMisesNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loc, conc = pred.chunk(2, dim=-1)            # 位置参数 loc 和浓度参数 conc
        conc = conc.clone()              # 克隆浓度参数 conc 以避免在原始张量上进行操作
        with torch.no_grad():            # 防止在计算中创建梯度
            conc.clamp_(min=self.eps)    # clamp_ 方法将浓度参数限制在 eps 以上
        nll = -conc * torch.cos(target - loc) + math.log(2 * math.pi) + _log_modified_bessel_fn(conc, order=0)    # 计算负对数似然损失
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))
