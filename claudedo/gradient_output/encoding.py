import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FreqEncoder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
    
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim

        self.output_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input, **kwargs):

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))

        out = torch.cat(out, dim=-1)


        return out

def get_encoder(encoding, input_dim=3, 
                multires=6, #6
                degree=4,
                num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=2048, align_corners=False,
                **kwargs):

    if encoding == 'None':
        return lambda x, **kwargs: x, input_dim

    elif encoding == 'Fourier':
        def encode(inputs, **kwargs):
            B, D = inputs.shape
            device = inputs.device

            # Original input retained
            out = [inputs]

            # Frequencies: 2^0, 2^1, ..., 2^{deg - 1}
            for i in range(multires):
                freq = 2 ** i
                scaled = inputs * freq

                # sin and cos via phase-shifted sin
                sin = torch.sin(scaled)
                cos = torch.sin(scaled + math.pi / 2)

                out.append(sin)
                out.append(cos)

            return torch.cat(out, dim=-1)

        out_dim = input_dim * (2 * multires + 1)
        return encode, out_dim

    elif encoding == 'Hash':
        from newhashencoder.hash_encoding import HashEmbedder
        bounding_box = (torch.tensor(-1)-torch.tensor([1.0,1.0,1.0]), torch.tensor(1)+torch.tensor([1.0,1.0,1.0]))
        encoder = HashEmbedder(bounding_box=bounding_box,
                             log2_hashmap_size=log2_hashmap_size,
                             finest_resolution=desired_resolution)

    else:
        raise NotImplementedError('Unknown encoding mode, choose from [None, frequency, sphere_harmonics, hashgrid, tiledgrid]')

    return encoder, encoder.output_dim