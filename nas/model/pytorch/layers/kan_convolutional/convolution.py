# Credits to: https://github.com/detkov/Convolution-From-Scratch/
from copy import deepcopy

import torch
import numpy as np
from typing import List, Tuple, Union


def calc_out_dims(matrix, kernel_side, stride, dilation, padding):
    def convert_to_int(t):
        if isinstance(t, np.ndarray):
            return t.astype(int)
        return t.to(torch.int)

    batch_size, n_channels, n, m = matrix.shape
    h_out = int(np.floor((n + 2 * padding[0] - kernel_side - (kernel_side - 1) * (dilation[0] - 1)) / stride[0])) + 1
    w_out = int(np.floor((m + 2 * padding[1] - kernel_side - (kernel_side - 1) * (dilation[1] - 1)) / stride[1])) + 1
    b = [kernel_side // 2, kernel_side // 2]
    return h_out, w_out, batch_size, n_channels


def kan_conv2d(matrix: Union[List[List[float]], np.ndarray],
               # but as torch tensors. Kernel side asume q el kernel es cuadrado
               kernel,
               kernel_side,
               stride=(1, 1),
               dilation=(1, 1),
               padding=(0, 0),
               device="cuda"
               ) -> torch.Tensor:
    """Makes a 2D convolution with the kernel over matrix using defined stride, dilation and padding along axes.

    Args:
        matrix (batch_size, colors, n, m]): 2D matrix to be convolved.
        kernel  (function]): 2D odd-shaped matrix (e.g. 3x3, 5x5, 13x9, etc.).
        stride (Tuple[int, int], optional): Tuple of the stride along axes. With the `(r, c)` stride we move on `r` pixels along rows and on `c` pixels along columns on each iteration. Defaults to (1, 1).
        dilation (Tuple[int, int], optional): Tuple of the dilation along axes. With the `(r, c)` dilation we distancing adjacent pixels in kernel by `r` along rows and `c` along columns. Defaults to (1, 1).
        padding (Tuple[int, int], optional): Tuple with number of rows and columns to be padded. Defaults to (0, 0).

    Returns:
        np.ndarray: 2D Feature map, i.e. matrix after convolution.
    """
    h_out, w_out, batch_size, n_channels = calc_out_dims(matrix, kernel_side, stride, dilation, padding)

    matrix_out = torch.zeros((batch_size, n_channels, h_out, w_out)).to(
        device)  # estamos asumiendo que no existe la dimension de rgb
    unfold = torch.nn.Unfold((kernel_side, kernel_side), dilation=dilation, padding=padding, stride=stride)

    for channel in range(n_channels):
        # print(matrix[:,channel,:,:].unsqueeze(1).shape)
        conv_groups = unfold(matrix[:, channel, :, :].unsqueeze(1)).transpose(1, 2)
        # print("conv",conv_groups.shape)
        for k in range(batch_size):
            matrix_out[k, channel, :, :] = kernel.forward(conv_groups[k, :, :]).reshape((h_out, w_out))
    return matrix_out


def multiple_convs_kan_conv2d(matrix,  # but as torch tensors. Kernel side asume q el kernel es cuadrado
                              kernels,
                              kernel_side,
                              stride=(1, 1),
                              dilation=(1, 1),
                              padding=(0, 0),
                              device="cuda"
                              ) -> torch.Tensor:
    """Makes a 2D convolution with the kernel over matrix using defined stride, dilation and padding along axes.

    Args:
        matrix (batch_size, colors, n, m]): 2D matrix to be convolved.
        kernel  (function]): 2D odd-shaped matrix (e.g. 3x3, 5x5, 13x9, etc.).
        stride (Tuple[int, int], optional): Tuple of the stride along axes. With the `(r, c)` stride we move on `r` pixels along rows and on `c` pixels along columns on each iteration. Defaults to (1, 1).
        dilation (Tuple[int, int], optional): Tuple of the dilation along axes. With the `(r, c)` dilation we distancing adjacent pixels in kernel by `r` along rows and `c` along columns. Defaults to (1, 1).
        padding (Tuple[int, int], optional): Tuple with number of rows and columns to be padded. Defaults to (0, 0).

    Returns:
        np.ndarray: 2D Feature map, i.e. matrix after convolution.
    """
    h_out, w_out, batch_size, n_channels = calc_out_dims(matrix, kernel_side, stride, dilation, padding)
    n_convs = len(kernels)
    matrix_out = torch.zeros((batch_size, n_channels * n_convs, h_out, w_out)).to(
        device)  # estamos asumiendo que no existe la dimension de rgb
    unfold = torch.nn.Unfold((kernel_side, kernel_side), dilation=dilation, padding=padding, stride=stride)
    conv_groups = unfold(matrix[:, :, :, :]).view(batch_size, n_channels, kernel_side * kernel_side,
                                                  h_out * w_out).transpose(2,
                                                                           3)  # reshape((batch_size,n_channels,h_out,w_out))

    for channel in range(n_channels):
        for kern in range(n_convs):
            matrix_out[:, kern + channel * n_convs, :, :] = kernels[kern].conv.forward(
                conv_groups[:, channel, :, :].flatten(0, 1)).reshape((batch_size, h_out, w_out))
    return matrix_out


def deep_conv_kan_conv2d(matrix,  # but as torch tensors. Kernel side asume q el kernel es cuadrado
                         kernel,
                         kernel_side,
                         stride=(1, 1),
                         dilation=(1, 1),
                         padding=(0, 0),
                         device="cuda"
                         ) -> torch.Tensor:
    """Makes a 2D convolution with the kernel over matrix using defined stride, dilation and padding along axes.

    Args:
        matrix (batch_size, colors, n, m]): 2D matrix to be convolved.
        kernel  (KAN_CrossConvolution): in×out channels of 2D odd-shaped matrices (e.g. 3x3, 5x5, 13x9, etc.).
        stride (Tuple[int, int], optional): Tuple of the stride along axes. With the `(r, c)` stride we move on `r` pixels along rows and on `c` pixels along columns on each iteration. Defaults to (1, 1).
        dilation (Tuple[int, int], optional): Tuple of the dilation along axes. With the `(r, c)` dilation we distancing adjacent pixels in kernel by `r` along rows and `c` along columns. Defaults to (1, 1).
        padding (Tuple[int, int], optional): Tuple with number of rows and columns to be padded. Defaults to (0, 0).

    Returns:
        np.ndarray: 2D Feature map, i.e. matrix after convolution.
    """
    h_out, w_out, batch_size, n_channels = calc_out_dims(matrix, kernel_side, stride, dilation, padding)
    in_channels, out_channels = kernel.in_channels, kernel.out_channels
    matrix_out = torch.zeros((batch_size, out_channels, h_out, w_out)).to(device)
    unfold = torch.nn.Unfold((kernel_side, kernel_side), dilation=dilation, padding=padding, stride=stride)
    # conv_groups :: (batch_size, input_channels, num_of_blocks, num_of_inputs_of_one_elementary_conv_operation)
    conv_groups = unfold(matrix[:, :, :, :]).view(batch_size, n_channels, kernel_side * kernel_side,
                                                  h_out * w_out).transpose(2, 3)

    # deep_kernel_groups :: (batch_size, h' * w', c * k_h * k_w)
    deep_kernel_groups = (conv_groups
                          .transpose(1, 2)  # Move n_channels to the end
                          .flatten(2, 3))  # Merge input channels with kernel folds

    for_transformation = deep_kernel_groups.flatten(0, 1)  # (batch_size * h' * w', c * k_h * k_w)

    transformed = (kernel.conv.forward(for_transformation)  # (batch_size * h' * w', c_out)
                   .reshape(batch_size, h_out, w_out, out_channels))

    featuremap_shaped = transformed.permute(0, 3, 1, 2)  # (batch_size, out_channels, h_out, w_out)
    return featuremap_shaped


class LinearKernel:
    """
    A wrapper around convolution without bias parameter
    """
    def __init__(self, in_channels, out_channels, kernel_side):
        self.conv = torch.nn.Linear(in_channels * kernel_side * kernel_side, out_channels, bias=False)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_side = kernel_side

    def flipped(self):
        """
        Flips along h and w axes
        """
        res = deepcopy(self)
        shaped = res.kernel_structured()
        flipped = torch.flip(shaped, [1, 2])
        res.conv.weight.data = flipped.reshape(res.out_channels, res.in_channels * res.kernel_side * res.kernel_side)
        return res

    def kernel_structured(self):
        t = self.conv.weight.data
        return t.reshape(self.in_channels * self.out_channels, self.kernel_side, self.kernel_side)

    def torch_structured(self):
        return self.conv.weight.data.reshape(self.out_channels, self.in_channels, self.kernel_side,
                                             self.kernel_side).transpose(0, 1)# .transpose(2, 3)

    @staticmethod
    def random_kernel(in_channels, out_channels, kernel_side):
        kernel = LinearKernel(in_channels, out_channels, kernel_side)
        kernel.conv.weight.data = torch.rand(out_channels, in_channels * kernel_side * kernel_side)
        return kernel


def transposed_kan_conv2d(matrix,  # but as torch tensors. Kernel side asume q el kernel es cuadrado
                          kernel,
                          kernel_side,
                          device="cuda"
                          ) -> torch.Tensor:
    """Makes a 2D convolution with the kernel over matrix using defined stride, dilation and padding along axes.

    Args:
        matrix (batch_size, colors, n, m]): 2D matrix to be convolved.
        kernel  (KAN_CrossConvolution | LinearKernel): in×out channels of 2D odd-shaped matrices (e.g. 3x3, 5x5, 13x9, etc.).

    Returns:
        np.ndarray: 2D Feature map, i.e. matrix after convolution.
    """
    return deep_conv_kan_conv2d(matrix, kernel, kernel_side, padding=(kernel_side - 1, kernel_side - 1), device=device)


def add_padding(matrix: np.ndarray,
                padding: Tuple[int, int]) -> np.ndarray:
    """Adds padding to the matrix. 

    Args:
        matrix (np.ndarray): Matrix that needs to be padded. Type is List[List[float]] casted to np.ndarray.
        padding (Tuple[int, int]): Tuple with number of rows and columns to be padded. With the `(r, c)` padding we addding `r` rows to the top and bottom and `c` columns to the left and to the right of the matrix

    Returns:
        np.ndarray: Padded matrix with shape `n + 2 * r, m + 2 * c`.
    """
    n, m = matrix.shape
    r, c = padding

    padded_matrix = np.zeros((n + r * 2, m + c * 2))
    padded_matrix[r: n + r, c: m + c] = matrix

    return padded_matrix
