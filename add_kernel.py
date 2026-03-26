import nki
import nki.language as nl
import nki.isa as nisa

@nki.jit
def nki_tensor_add_kernel(a_input, b_input):
    """
    nki kernel for elementwise addition of two kernels
    """
    # ensures both a_input and b_input are the same shape
    assert a_input.shape == b_input.shape
    # ensures a_input fits without tiling
    assert a_input.shape[0] <= nl.tile_size.pmax

    # Allocate space in SBUF for input tensors
    a_tile = sbuf.view(dtype=a_input.dtype, shape = a_input.shape)
    b_tile = sbuf.view(dtype=b_input.dtype, shape = b_input.shape)

    # Copy from HBM to SBUF
    nisa.dma_copy(dst = a_tile, src = a_input)
    nisa.dma_copy(dst = b_tile, src = b_input)

    # Allocate space for the output tensor
    out_tile = sbuf.view(dtype = a_input.dtype, shape = a_input.shape)
    nisa.tensor_tensor(dst = out_tile, data1 = a_tile, data2 = b_tile, op = nl.add)

    out_tensor = hbm.view(dtype = a_input.dtype, shape = a_input.shape)
    nisa.dma_copy(dst = out_tensor, src = out_tile)

    return out_tensor






