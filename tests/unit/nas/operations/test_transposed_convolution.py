from copy import deepcopy

import torch
import torch.nn.functional as F

from nas.model.pytorch.layers.kan_convolutional.convolution import transposed_kan_conv2d, LinearKernel


def test_transposed_conv2d():
    in_channels = 10
    out_channels = 11
    kernel = LinearKernel(in_channels, out_channels, 3)
    # print(kernel.conv.weight.data.shape)
    # kernel.conv.weight.data = torch.Tensor([list(range(1, 10))])
    structured = kernel.torch_structured()
    # print(structured)

    h = 8
    w = 9
    sample_input = torch.randn(1, in_channels, h, w)
    # sample_input = torch.tensor([[[[1.0]]]])
    torch_output = F.conv_transpose2d(sample_input, structured)
    print(torch_output.shape)
    nas_output = transposed_kan_conv2d(sample_input, kernel.flipped(), 3)
    print(nas_output.shape)

    print(torch_output[0][0])
    print(nas_output[0][0])

    assert torch.allclose(torch_output, nas_output, atol=1e-5, rtol=1e-5)
