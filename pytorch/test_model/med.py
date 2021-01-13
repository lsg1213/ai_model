
import torch


def medfilt (x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert len(x.size()) == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = torch.zeros ((len (x), k), dtype=x.dtype, device=x.device)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return torch.median (y, dim=1)[0]


def test ():
    import pylab as p
    x = torch.linspace (0, 1, 101)
    x[3::10] = 1.5
    p.plot (x)
    p.plot (medfilt (x,3).numpy())
    p.plot (medfilt (x,5).numpy())
    p.show ()


if __name__ == '__main__':
    test ()