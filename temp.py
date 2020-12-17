import torch

from Modules import FFT_Block

layer = FFT_Block(
    in_channels= 384,
    heads= 4,
    dropout_rate= 0.1,
    fft_in_kernel_size= 3,
    fft_out_kernel_size= 1,
    fft_channels= 1536
    )

x = torch.randn(3, 384, 523)
out = layer(x)

print(out.shape)