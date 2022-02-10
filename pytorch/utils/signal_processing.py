import torch
import torch.nn.functional as F


def lfilter_in_freq(input: torch.Tensor, filter: torch.Tensor):
    fN = input.shape[-1] // 2
    odd = input.shape[-1] % 2 == 1
    filter = torch.fft.fft(F.pad(filter, (0, input.shape[-1] - filter.shape[-1])))
    input = torch.fft.fft(input)
    res = input * filter
    if odd:
        fN += 1
        res = torch.cat([res[...,:fN], res[...,1:fN].flip(-1).conj()], -1)
    else:
        res = torch.cat([res[...,:fN+1], res[...,1:fN].flip(-1).conj()], -1)
    return torch.fft.ifft(res).real

