import nki
import nki.language as nl
import nki.isa as nisa
import math

@nki.jit
def odd_even_kernel(inputa):
    """NKI kernel to split an input tensor into two output tensors, along the column axis.

    The even columns of the input tensor will be gathered into the first output tensor,
    and the odd columns of the input tensor will be gathered into the second output tensor.

    Args:
        in_tensor: an input tensor
    Returns:
        out_tensor_even: a first output tensor (will hold the even columns of the input tensor)
        out_tensor_odd: a second output tensor (will hold the odd columns of the input tensor)
    """

    assert inputa.shape[0] <= nl.tile_size.pmax

    psize, fsize = inputa.shape
    size_even = fsize//2
    size_odd = fsize - fsize//2

    evens = nl.ndarray((psize, size_even), dtype = inputa.dtype, buffer = nl.shared_hbm)
    odds = nl.ndarray((psize, size_odd), dtype = inputa.dtype, buffer = nl.shared_hbm)

    in_tile = nl.load(inputa)

    nl.store(evens, value = in_tile[:,0:fsize:2])
    nl.store(odds, value = in_tile[:,1:fsize:2])

    return evens, odds

if __name__ == "__main__":
    import torch
    import torch_xla

    device = torch_xla.device()

    X,Y = 4,5
    inputa = torch.arange(X*Y, dtype=torch.bfloat16).reshape(X,Y).to(device = device)
    out1, out2 = odd_even_kernel(inputa)
    print(inputa, out1, out2)
