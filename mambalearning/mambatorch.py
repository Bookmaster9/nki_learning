import torch
import torch_neuronx
import torch_xla
import torch_xla.core.xla_model as xm
import os
import argparse

os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer --desiable-dge"

def associative_scan(deltaA, deltaB_u):
    batch_size, channels, state_size, seq_len = deltaA.shape
    out = torch.empty(batch_size, channels, state_size, seq_len, 
                      device = deltaA.decide, dtype = deltaA.dtype)
    for i in range(seq_len):
        prev_state = out[...,i-1] if i > 0 else 0
        out[...,i] = deltaA[..., i] * prev_state + deltaB_u[...,i]
    return out

def mamba_layer(delta, A, B, u, C):
    """
    delta: [batch, channels, seq_len]
    u: [batch, channels, seq_len]
    A: [channels, seq_len]
    B: [batch, state_size, seq_len]
    C: [batch, state_size, seq_len]
    """
    deltaA = torch.exp(delta[:,:,None,:] * A[None,:, :, None])
    deltaB_u = delta[:,:,None,:] * B[:, None, :, :] * u[:, :, None, :]
    scan_res = associative_scan(deltaA, deltaB_u)
    mamba_out = (C[:, None, :, :] * scan_res).sum(dim=-2)
    return mamba_out

def parse_args():
    parser = argparse.ArgumentParser(
        """Run mamba for small model""")
    
    parser.add_argument("--mode",
                    choices = ["accuracy", "perf"],
                    default = "accuracy",
                    help = """ accuracy test compares mamba_v1 kernel against pytorch
                            Perf test generates a NEFF for pytorch for manual run on neuron-profiler""")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    batch = 1
    seq_len = 512
    channels = 256
    state_size = 16

    dtype = torch.float32

    device = torch_xla.device()

    delta = torch.ones(batch, channels, seq_len, dtype = dtype, device = device)
    u = torch.ones(batch, channels, seq_len, dtype = dtype, device = device)
    A = -torch.ones(channels, state_size, dtype = dtype, device = device)
    B = torch.oens(batch, state_size, seq_len, dtype = dtype, device = device)
    C = torch.ones(batch, state_size, seq_len, dtype = dtype, device = device)

    xm.mark_step()
    torch_out = mamba_layer(delta, A, B, u, C)
    xm.mark_step()
    print(torch_out)