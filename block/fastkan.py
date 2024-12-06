# import torch
# import torch.nn as nn


# class RadialBasisFunction(nn.Module):
#     def __init__(
#             self,
#             grid_min: float = -2.,
#             grid_max: float = 2.,
#             num_grids: int = 8,
#             denominator: float = None,  # larger denominators lead to smoother basis
#     ):
#         super().__init__()
#         grid = torch.linspace(grid_min, grid_max, num_grids)
#         self.grid = torch.nn.Parameter(grid, requires_grad=False)
#         self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

#     def forward(self, x):
#         return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


# class FastKANConvNDLayer(nn.Module):
#     def __init__(self, conv_class, norm_class, input_dim, output_dim, kernel_size,
#                  groups=1, padding=0, stride=1, dilation=1,
#                  ndim: int = 2, grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0):
#         super(FastKANConvNDLayer, self).__init__()
#         self.inputdim = input_dim
#         self.outdim = output_dim
#         self.kernel_size = kernel_size
#         self.padding = padding
#         self.stride = stride
#         self.dilation = dilation
#         self.groups = groups
#         self.ndim = ndim
#         self.grid_size = grid_size
#         self.base_activation = base_activation()
#         self.grid_range = grid_range

#         if groups <= 0:
#             raise ValueError('groups must be a positive integer')
#         if input_dim % groups != 0:
#             raise ValueError('input_dim must be divisible by groups')
#         if output_dim % groups != 0:
#             raise ValueError('output_dim must be divisible by groups')

#         self.base_conv = nn.ModuleList([conv_class(input_dim // groups,
#                                                    output_dim // groups,
#                                                    kernel_size,
#                                                    stride,
#                                                    padding,
#                                                    dilation,
#                                                    groups=1,
#                                                    bias=False) for _ in range(groups)])

#         self.spline_conv = nn.ModuleList([conv_class(grid_size * input_dim // groups,
#                                                      output_dim // groups,
#                                                      kernel_size,
#                                                      stride,
#                                                      padding,
#                                                      dilation,
#                                                      groups=1,
#                                                      bias=False) for _ in range(groups)])

#         self.layer_norm = nn.ModuleList([norm_class(output_dim // groups) for _ in range(groups)])

#         self.rbf = RadialBasisFunction(grid_range[0], grid_range[1], grid_size)

#         self.dropout = None
#         if dropout > 0:
#             if ndim == 1:
#                 self.dropout = nn.Dropout1d(p=dropout)
#             if ndim == 2:
#                 self.dropout = nn.Dropout2d(p=dropout)
#             if ndim == 3:
#                 self.dropout = nn.Dropout3d(p=dropout)

#         # Initialize weights using Kaiming uniform distribution for better training start
#         for conv_layer in self.base_conv:
#             nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

#         for conv_layer in self.spline_conv:
#             nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

#     def forward_fast_kan(self, x, group_index):

#         # Apply base activation to input and then linear transform with base weights
#         base_output = self.base_conv[group_index](self.base_activation(x))
#         if self.dropout is not None:
#             x = self.dropout(x)
#         spline_basis = self.rbf(self.layer_norm[group_index](x))
#         spline_basis = spline_basis.moveaxis(-1, 2).flatten(1, 2)
#         spline_output = self.spline_conv[group_index](spline_basis)
#         x = base_output + spline_output



#         return x

#     def forward(self, x):
#         split_x = torch.split(x, self.inputdim // self.groups, dim=1)
#         output = []
#         for group_ind, _x in enumerate(split_x):
#             y = self.forward_fast_kan(_x.clone(), group_ind)
#             output.append(y.clone())
#         y = torch.cat(output, dim=1)
#         return y

#     def get_param_count(self):
#         total_params = sum(p.numel() for p in self.parameters())
#         return total_params


# class FastKANConv3DLayer(FastKANConvNDLayer):
#     def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
#                  grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0):
#         super(FastKANConv3DLayer, self).__init__(nn.Conv3d, nn.InstanceNorm3d,
#                                                  input_dim, output_dim,
#                                                  kernel_size,
#                                                  groups=groups, padding=padding, stride=stride, dilation=dilation,
#                                                  ndim=3,
#                                                  grid_size=grid_size, base_activation=base_activation,
#                                                  grid_range=grid_range,
#                                                  dropout=dropout)


# class FastKANConv2DLayer(FastKANConvNDLayer):
#     def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
#                  grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0):
#         super(FastKANConv2DLayer, self).__init__(nn.Conv2d, nn.InstanceNorm2d,
#                                                  input_dim, output_dim,
#                                                  kernel_size,
#                                                  groups=groups, padding=padding, stride=stride, dilation=dilation,
#                                                  ndim=2,
#                                                  grid_size=grid_size, base_activation=base_activation,
#                                                  grid_range=grid_range,
#                                                  dropout=dropout)

# # class FastKANConv2DLayer(FastKANConvNDLayer):
# #     def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
# #                  grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0):
# #         super(FastKANConv2DLayer, self).__init__(nn.Conv2d, nn.InstanceNorm2d,
# #                                                  input_dim, output_dim,
# #                                                  kernel_size,
# #                                                  groups=groups, padding=padding, stride=stride, dilation=dilation,
# #                                                  ndim=2,
# #                                                  grid_size=grid_size, base_activation=base_activation,
# #                                                  grid_range=grid_range,
# #                                                  dropout=dropout)




# class FastKANConv1DLayer(FastKANConvNDLayer):
#     def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
#                  grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0):
#         super(FastKANConv1DLayer, self).__init__(nn.Conv1d, nn.InstanceNorm1d,
#                                                  input_dim, output_dim,
#                                                  kernel_size,
#                                                  groups=groups, padding=padding, stride=stride, dilation=dilation,
#                                                  ndim=1,
#                                                  grid_size=grid_size, base_activation=base_activation,
#                                                  grid_range=grid_range,
#                                                  dropout=dropout)

# #测试1d卷积
# input_tensor_1d = torch.rand(16, 4, 10000)  # (batch_size, channels, width)
# conv1d = FastKANConv1DLayer(input_dim=4, output_dim=32, kernel_size=3, groups=4, padding=1)
# output_1d = conv1d(input_tensor_1d)
# print("1D Output shape:", output_1d.shape)  # 预期输出: torch.Size([1, 32, 50])

# #测试2D卷积
# input_tensor_2d = torch.rand(1, 16, 50, 50)  # (batch_size, channels, height, width)
# conv2d = FastKANConv2DLayer(input_dim=16, output_dim=100, kernel_size=3, groups=4, padding=1)
# output_2d = conv2d(input_tensor_2d)
# print("2D Output shape:", output_2d.shape)  # 预期输出: torch.Size([1, 32, 50, 50])
# print(f"Total parameters in FastKANConv2DLayer: {conv2d.get_param_count()}")

# # 测试3D卷积
# input_tensor_3d = torch.rand(1, 16, 20, 50, 50)  # (batch_size, channels, depth, height, width)
# conv3d = FastKANConv3DLayer(input_dim=16, output_dim=32, kernel_size=3, groups=4, padding=1)
# output_3d = conv3d(input_tensor_3d)
# print("3D Output shape:", output_3d.shape)
#!!!!!!!!!!!!!以上是原始 fastkan ！！！！！！！！！！！！！！！！


#
# #先升后降fastkan
import torch
import torch.nn as nn

class RadialBasisFunction(nn.Module):
    def __init__(
            self,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)
class FastKANConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_class, input_dim, output_dim, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1,
                 ndim: int = 2, grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0):
        super(FastKANConvNDLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.ndim = ndim
        self.grid_size = grid_size
        self.base_activation = base_activation()
        self.grid_range = grid_range

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        # 基础卷积层，用于生成 base_output
        self.base_conv = nn.ModuleList([
            conv_class(input_dim // groups, output_dim // groups, kernel_size,
                       stride, padding, dilation, groups=1, bias=False)
            for _ in range(groups)
        ])

        # **新增升维卷积层**，将输入升维到 4 倍的通道数
        self.expand_conv = nn.ModuleList([
            conv_class(input_dim // groups, 4 * input_dim // groups, 1,
                       bias=False) for _ in range(groups)
        ])



        # 层归一化与 RBF 层
        self.layer_norm = nn.ModuleList([norm_class(4 * input_dim // groups) for _ in range(groups)])
        self.rbf = RadialBasisFunction(grid_range[0], grid_range[1], grid_size)

        # **新增降维卷积层**，将经过 RBF 的特征降维回原始维度
        self.reduce_conv = nn.ModuleList([
            conv_class(grid_size * 4 * input_dim // groups, output_dim // groups, 1,
                       bias=False) for _ in range(groups)
        ])


        # Dropout层选择
        self.dropout = None
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            elif ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            elif ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)

        # 权重初始化
        for conv_layer in self.base_conv + self.expand_conv + self.reduce_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

    def forward_fast_kan(self, x, group_index):
        # 基础卷积层的输出
        base_output = self.base_conv[group_index](self.base_activation(x))

        if self.dropout is not None:
            x = self.dropout(x)

        # 1. **升维处理**
        x = self.expand_conv[group_index](x)  # [B, 4*input_dim, H, W]

        # 2. **归一化 + RBF**
        spline_basis = self.rbf(self.layer_norm[group_index](x))
        spline_basis = spline_basis.moveaxis(-1, 2).flatten(1, 2)

        # 3. **降维处理**
        spline_output = self.reduce_conv[group_index](spline_basis)  # [B, output_dim, H, W]

        # 4. **相加形成最终输出**
        x = base_output + spline_output

        return x

    def forward(self, x):
        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_fast_kan(_x.clone(), group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y

    def get_param_count(self):
        total_params = sum(p.numel() for p in self.parameters())
        return total_params


class FastKANConv3DLayer(FastKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0):
        super(FastKANConv3DLayer, self).__init__(nn.Conv3d, nn.InstanceNorm3d,
                                                 input_dim, output_dim,
                                                 kernel_size,
                                                 groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                 ndim=3,
                                                 grid_size=grid_size, base_activation=base_activation,
                                                 grid_range=grid_range,
                                                 dropout=dropout)


class FastKANConv2DLayer(FastKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0):
        super(FastKANConv2DLayer, self).__init__(nn.Conv2d, nn.InstanceNorm2d,
                                                 input_dim, output_dim,
                                                 kernel_size,
                                                 groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                 ndim=2,
                                                 grid_size=grid_size, base_activation=base_activation,
                                                 grid_range=grid_range,
                                                 dropout=dropout)

class FastKANConv2DLayer(FastKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0):
        super(FastKANConv2DLayer, self).__init__(nn.Conv2d, nn.InstanceNorm2d,
                                                 input_dim, output_dim,
                                                 kernel_size,
                                                 groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                 ndim=2,
                                                 grid_size=grid_size, base_activation=base_activation,
                                                 grid_range=grid_range,
                                                 dropout=dropout)




class FastKANConv1DLayer(FastKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0):
        super(FastKANConv1DLayer, self).__init__(nn.Conv1d, nn.InstanceNorm1d,
                                                 input_dim, output_dim,
                                                 kernel_size,
                                                 groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                 ndim=1,
                                                 grid_size=grid_size, base_activation=base_activation,
                                                 grid_range=grid_range,
                                                 dropout=dropout)

#测试1d卷积
input_tensor_1d = torch.rand(16, 4, 10000)  # (batch_size, channels, width)
conv1d = FastKANConv1DLayer(input_dim=4, output_dim=32, kernel_size=3, groups=4, padding=1)
output_1d = conv1d(input_tensor_1d)
print("1D Output shape:", output_1d.shape)  # 预期输出: torch.Size([1, 32, 50])


#测试2D卷积
input_tensor_2d = torch.rand(1, 16, 50, 50)  # (batch_size, channels, height, width)
conv2d = FastKANConv2DLayer(input_dim=16, output_dim=100, kernel_size=3, groups=4, padding=1)
output_2d = conv2d(input_tensor_2d)
print("2D Output shape:", output_2d.shape)  # 预期输出: torch.Size([1, 32, 50, 50])
print(f"Total parameters in FastKANConv2DLayer: {conv2d.get_param_count()}")

# 测试3D卷积
input_tensor_3d = torch.rand(1, 16, 20, 50, 50)  # (batch_size, channels, depth, height, width)
conv3d = FastKANConv3DLayer(input_dim=16, output_dim=32, kernel_size=3, groups=4, padding=1)
output_3d = conv3d(input_tensor_3d)
print("3D Output shape:", output_3d.shape)






#先 降维 再升维
#
# import torch
# import torch.nn as nn
#
# class RadialBasisFunction(nn.Module):
#     def __init__(
#             self,
#             grid_min: float = -2.,
#             grid_max: float = 2.,
#             num_grids: int = 8,
#             denominator: float = None,  # larger denominators lead to smoother basis
#     ):
#         super().__init__()
#         grid = torch.linspace(grid_min, grid_max, num_grids)
#         self.grid = torch.nn.Parameter(grid, requires_grad=False)
#         self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)
#
#     def forward(self, x):
#         return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)
# class FastKANConvNDLayer(nn.Module):
#     def __init__(self, conv_class, norm_class, input_dim, output_dim, kernel_size,
#                  groups=1, padding=0, stride=1, dilation=1,
#                  ndim: int = 2, grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0):
#         super(FastKANConvNDLayer, self).__init__()
#         self.inputdim = input_dim
#         self.outdim = output_dim
#         self.kernel_size = kernel_size
#         self.padding = padding
#         self.stride = stride
#         self.dilation = dilation
#         self.groups = groups
#         self.ndim = ndim
#         self.grid_size = grid_size
#         self.base_activation = base_activation()
#         self.grid_range = grid_range
#
#         if groups <= 0:
#             raise ValueError('groups must be a positive integer')
#         if input_dim % groups != 0:
#             raise ValueError('input_dim must be divisible by groups')
#         if output_dim % groups != 0:
#             raise ValueError('output_dim must be divisible by groups')
#
#         # 基础卷积层，用于生成 base_output
#         self.base_conv = nn.ModuleList([
#             conv_class(input_dim // groups, output_dim // groups, kernel_size,
#                        stride, padding, dilation, groups=1, bias=False)
#             for _ in range(groups)
#         ])
#
#         # 降维
#         self.expand_conv = nn.ModuleList([
#             conv_class(input_dim // groups,  input_dim // (groups*4), 1,
#                        bias=False) for _ in range(groups)
#         ])
#
#         # 增维
#         self.reduce_conv = nn.ModuleList([
#             conv_class(grid_size  * input_dim // (groups*4), output_dim  // groups, 1,
#                        bias=False) for _ in range(groups)
#         ])
#
#         # 层归一化与 RBF 层
#         self.layer_norm = nn.ModuleList([norm_class(4 * input_dim // groups) for _ in range(groups)])
#         self.rbf = RadialBasisFunction(grid_range[0], grid_range[1], grid_size)
#
#
#         # Dropout层选择
#         self.dropout = None
#         if dropout > 0:
#             if ndim == 1:
#                 self.dropout = nn.Dropout1d(p=dropout)
#             elif ndim == 2:
#                 self.dropout = nn.Dropout2d(p=dropout)
#             elif ndim == 3:
#                 self.dropout = nn.Dropout3d(p=dropout)
#
#         # 权重初始化
#         for conv_layer in self.base_conv + self.expand_conv + self.reduce_conv:
#             nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')
#
#     def forward_fast_kan(self, x, group_index):
#         # 基础卷积层的输出
#         base_output = self.base_conv[group_index](self.base_activation(x))
#
#         if self.dropout is not None:
#             x = self.dropout(x)
#
#         # 1. **降维处理**
#         x = self.expand_conv[group_index](x)  # [B, 4*input_dim, H, W]
#         print(x.shape)
#
#         # 2. **归一化 + RBF**
#         spline_basis = self.rbf(self.layer_norm[group_index](x))
#         spline_basis = spline_basis.moveaxis(-1, 2).flatten(1, 2)
#         print(spline_basis.shape)
#
#         # 3. **升维处理**
#         spline_output = self.reduce_conv[group_index](spline_basis)  # [B, output_dim, H, W]
#         print(spline_output.shape)
#
#         # 4. **相加形成最终输出**
#         x = base_output + spline_output
#
#         return x
#
#     def forward(self, x):
#         split_x = torch.split(x, self.inputdim // self.groups, dim=1)
#         output = []
#         for group_ind, _x in enumerate(split_x):
#             y = self.forward_fast_kan(_x.clone(), group_ind)
#             output.append(y.clone())
#         y = torch.cat(output, dim=1)
#         #print(y.shape)
#         return y
#
#
#
# class FastKANConv3DLayer(FastKANConvNDLayer):
#     def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
#                  grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0):
#         super(FastKANConv3DLayer, self).__init__(nn.Conv3d, nn.InstanceNorm3d,
#                                                  input_dim, output_dim,
#                                                  kernel_size,
#                                                  groups=groups, padding=padding, stride=stride, dilation=dilation,
#                                                  ndim=3,
#                                                  grid_size=grid_size, base_activation=base_activation,
#                                                  grid_range=grid_range,
#                                                  dropout=dropout)
#
#
# class FastKANConv2DLayer(FastKANConvNDLayer):
#     def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
#                  grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0):
#         super(FastKANConv2DLayer, self).__init__(nn.Conv2d, nn.InstanceNorm2d,
#                                                  input_dim, output_dim,
#                                                  kernel_size,
#                                                  groups=groups, padding=padding, stride=stride, dilation=dilation,
#                                                  ndim=2,
#                                                  grid_size=grid_size, base_activation=base_activation,
#                                                  grid_range=grid_range,
#                                                  dropout=dropout)
#
# # class FastKANConv2DLayer(FastKANConvNDLayer):
# #     def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
# #                  grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0):
# #         super(FastKANConv2DLayer, self).__init__(nn.Conv2d, nn.InstanceNorm2d,
# #                                                  input_dim, output_dim,
# #                                                  kernel_size,
# #                                                  groups=groups, padding=padding, stride=stride, dilation=dilation,
# #                                                  ndim=2,
# #                                                  grid_size=grid_size, base_activation=base_activation,
# #                                                  grid_range=grid_range,
# #                                                  dropout=dropout)
#
#
#
#
# class FastKANConv1DLayer3(FastKANConvNDLayer):
#     def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
#                  grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0):
#         super(FastKANConv1DLayer3, self).__init__(nn.Conv1d, nn.InstanceNorm1d,
#                                                  input_dim, output_dim,
#                                                  kernel_size,
#                                                  groups=groups, padding=padding, stride=stride, dilation=dilation,
#                                                  ndim=1,
#                                                  grid_size=grid_size, base_activation=base_activation,
#                                                  grid_range=grid_range,
#                                                  dropout=dropout)
#
# #测试1d卷积
# input_tensor_1d = torch.rand(16, 32, 10000)  # (batch_size, channels, width)
# conv1d = FastKANConv1DLayer3(input_dim=32, output_dim=128, kernel_size=3, groups=4, padding=1)
# output_1d = conv1d(input_tensor_1d)
# print("1D Output shape:", output_1d.shape)  # 预期输出: torch.Size([1, 32, 50])
#
# #测试2D卷积
# input_tensor_2d = torch.rand(1, 16, 50, 50)  # (batch_size, channels, height, width)
# conv2d = FastKANConv2DLayer(input_dim=16, output_dim=100, kernel_size=3, groups=4, padding=1)
# output_2d = conv2d(input_tensor_2d)
# print("2D Output shape:", output_2d.shape)  # 预期输出: torch.Size([1, 32, 50, 50])

# # 测试3D卷积
# input_tensor_3d = torch.rand(1, 16, 20, 50, 50)  # (batch_size, channels, depth, height, width)
# conv3d = FastKANConv3DLayer(input_dim=16, output_dim=32, kernel_size=3, groups=4, padding=1)
# output_3d = conv3d(input_tensor_3d)
# print("3D Output shape:", output_3d.shape)

