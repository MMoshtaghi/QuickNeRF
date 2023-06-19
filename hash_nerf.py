import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb



class HashNeRFDensityMLP(nn.Module):
    def __init__(self, input_ch, num_layers, hidden_size, output_ch):
        super().__init__()

        assert num_layers >= 2

        self.input_layer = nn.Linear(input_ch, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 2)])
        self.output_layer = nn.Linear(hidden_size, output_ch)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)
        x = self.output_layer(x)
        sigma = torch.exp(x[..., 0])

        return sigma, x


class HashNeRFColorMLP(nn.Module):
    def __init__(self, density_output_ch, input_ch_views, num_layers, hidden_size, output_ch=3):
        super().__init__()

        assert num_layers >= 2

        self.input_layer = nn.Linear(density_output_ch + input_ch_views, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 2)])
        self.output_layer = nn.Linear(hidden_size, output_ch)

    def forward(self, density_output, input_views):
        x = torch.cat([density_output, input_views], dim=-1)
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)
        x = self.output_layer(x)

        # color = torch.sigmoid(x)
        # color = torch.exp(x)
        color = x

        return color


class HashNeRF(nn.Module):
    """
    A smaller NeRF enough for nerf training with hash encoding
    """
    def __init__(self, num_layers_density=2, num_layers_color=3, 
                       hidden_size=64, density_output_ch=16,
                       input_ch=3, input_ch_views=3):
        super().__init__()

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views

        # Density MLP
        self.density_mlp = HashNeRFDensityMLP(input_ch, num_layers_density, hidden_size, density_output_ch)

        # Color MLP
        self.color_mlp = HashNeRFColorMLP(density_output_ch, input_ch_views, num_layers_color, hidden_size)


    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        sigma, density_output = self.density_mlp(input_pts)
        color = self.color_mlp(density_output, input_views)

        outputs = torch.cat([color, sigma.unsqueeze(dim=-1)], -1)

        return outputs



class HashEmbedding(nn.Module):
    def __init__(self, 
                 x_boundary,
                 n_levels=16, 
                 log2_hashmap_size=19, 
                 n_features_per_level=2,
                 coarest_resolution=16, 
                 finest_resolution=512):
        super().__init__()

        self.n_levels = n_levels
        self.log2_hashmap_size = log2_hashmap_size
        self.n_features_per_level = n_features_per_level
        self.coarest_resolution = coarest_resolution
        self.finest_resolution = finest_resolution
        self.out_size = self.n_levels * self.n_features_per_level

        self.b = np.exp((np.log(self.finest_resolution) - np.log(self.coarest_resolution)) / (n_levels - 1))

        self.embeddings = nn.ModuleList([nn.Embedding(2 ** self.log2_hashmap_size, self.n_features_per_level) 
                                         for i in range(n_levels)])

        self.x_min = x_boundary[0]
        self.x_max = x_boundary[1]

        self.cube_offsets = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]])

        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)

    def hash_function(self, voxel_all_vertices):
        primes = [1, 2654435761, 805459861]
        dim = voxel_all_vertices.shape[-1]
        assert dim == 3

        xor_result = torch.zeros_like(voxel_all_vertices)[..., 0]
        for i in range(3):
            xor_result = xor_result ^ voxel_all_vertices[..., i] * primes[i]

        hash_indices = xor_result % (2 ** self.log2_hashmap_size)
        return hash_indices

    def total_variation_loss(self, level=0):
        # Get resolution
        resolution = torch.tensor(np.floor(self.coarest_resolution * (self.b ** level))).int()
    
        # Cube size to apply TV loss
        min_cube_size = self.coarest_resolution - 1
        max_cube_size = 50 # can be tuned
        cube_size = torch.floor(torch.clip(resolution / 10.0, min_cube_size, max_cube_size)).int()
    
        # Sample cuboid
        min_vertex = torch.randint(0, resolution - cube_size, (3,))
        idx = min_vertex + torch.stack([torch.arange(cube_size + 1) for _ in range(3)], dim=-1)
        cube_indices = torch.stack(torch.meshgrid(idx[:,0], idx[:,1], idx[:,2]), dim=-1)
    
        hashed_indices = self.hash_function(cube_indices)
        cube_embeddings = self.embeddings[level](hashed_indices)
    
        # Compute loss
        tv_x = torch.pow(cube_embeddings[1:,:,:,:]-cube_embeddings[:-1,:,:,:], 2).sum()
        tv_y = torch.pow(cube_embeddings[:,1:,:,:]-cube_embeddings[:,:-1,:,:], 2).sum()
        tv_z = torch.pow(cube_embeddings[:,:,1:,:]-cube_embeddings[:,:,:-1,:], 2).sum()
    
        return (tv_x + tv_y + tv_z) / cube_size



    def trilinear_interpolate(self, x, voxel_min_vertices, voxel_embeddings):

        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = x - voxel_min_vertices

        # step 0
        c000 = voxel_embeddings[:, 0]
        c001 = voxel_embeddings[:, 1]
        c010 = voxel_embeddings[:, 2]
        c011 = voxel_embeddings[:, 3]
        c100 = voxel_embeddings[:, 4]
        c101 = voxel_embeddings[:, 5]
        c110 = voxel_embeddings[:, 6]
        c111 = voxel_embeddings[:, 7]

        # step 1
        x_d = weights[:, 0].unsqueeze(-1)
        c00 = c000 * (1 - x_d) + c100 * x_d 
        c01 = c001 * (1 - x_d) + c101 * x_d 
        c10 = c010 * (1 - x_d) + c110 * x_d 
        c11 = c011 * (1 - x_d) + c111 * x_d 

        # step 2
        y_d = weights[:, 1].unsqueeze(-1)
        c0 = c00 * (1 - y_d) + c10 * y_d 
        c1 = c01 * (1 - y_d) + c11 * y_d 

        # step 3
        z_d = weights[:, 2].unsqueeze(-1)
        c = c0 * (1 - z_d) + c1 * z_d

        return c

    def forward(self, x):

        ## update min and max for scaling
        #if self.training:
        #    self.x_min = torch.min(self.x_min, x.min(dim=0)[0])
        #    self.x_max = torch.max(self.x_max, x.max(dim=0)[0])
        
        all_level_embeddings = []
        x_min = self.x_min
        x_max = self.x_max

        for l in range(self.n_levels):
            # get resolution and scale
            resolution = np.floor(self.coarest_resolution * (self.b ** l))
            scaled_x = (x - x_min) / (x_max - x_min) * resolution

            # get vertices
            voxel_min_vertices = torch.floor(scaled_x).int()
            voxel_all_vertices = voxel_min_vertices.unsqueeze(1) + self.cube_offsets
            
            # get hash indice and embeddings
            voxel_hash_indices = self.hash_function(voxel_all_vertices)
            voxel_embeddings = self.embeddings[l](voxel_hash_indices)
            
            # interpolate
            level_embeddings = self.trilinear_interpolate(scaled_x, voxel_min_vertices, voxel_embeddings)
            all_level_embeddings.append(level_embeddings)

        x_embedding = torch.cat(all_level_embeddings, dim=-1)
        return x_embedding


class SHEncoder(nn.Module):
    def __init__(self, input_dim=3, degree=4):
    
        super().__init__()

        self.input_dim = input_dim
        self.degree = degree

        assert self.input_dim == 3
        assert self.degree >= 1 and self.degree <= 5

        self.out_size = degree ** 2

        self.C0 = 0.28209479177387814
        self.C1 = 0.4886025119029199
        self.C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        self.C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]
        self.C4 = [
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761
        ]

    def forward(self, input, **kwargs):

        result = torch.empty((*input.shape[:-1], self.out_size), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)

        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                #result[..., 6] = self.C2[2] * (3.0 * zz - 1) # xx + yy + zz == 1, but this will lead to different backward gradients, interesting...
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        return result
