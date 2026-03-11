"""
Created on Mon Jul 28 15:54:29 2025

@author: Bangchen Yin
"""
import jax
import jax.numpy as jnp
from jax import random, lax, vmap
import haiku as hk

from typing import Optional, Tuple, List, NamedTuple, Any
import math
from functools import partial

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
def segment_mean(data, segment_ids, num_segments):
    segment_ids = segment_ids.astype(jnp.int32)
    ones = jnp.ones((data.shape[0], *((1,) * (data.ndim - 1))), dtype=data.dtype)
    segment_count = jax.ops.segment_sum(ones, segment_ids, num_segments)
    segment_sum = jax.ops.segment_sum(data, segment_ids, num_segments)
    segment_count = jnp.where(segment_count == 0, 1, segment_count)
    return segment_sum / segment_count

class rbf_emb(hk.Module):
    def __init__(self, num_basis=8, r_max=5.0, trainable=True, name=None):
        super().__init__(name=name)
        self.num_basis = num_basis
        self.r_max = r_max
        self.trainable = trainable
        
    def __call__(self, x):
        prefactor = 2.0 / self.r_max
        init_values = jnp.pi * jnp.arange(1, self.num_basis + 1)
        
        if self.trainable:
            bessel_weights = hk.get_parameter(
                'bessel_weights',
                shape=(self.num_basis,),
                init=hk.initializers.Constant(init_values)
            )
        else:
            bessel_weights = init_values
        
        x_expanded = x[..., jnp.newaxis]
        numerator = jnp.sin(bessel_weights * x_expanded / self.r_max)
        result = prefactor * (numerator / x_expanded)
        return result

class NeighborEmb(hk.Module):
    def __init__(self, hid_dim, name=None):
        super().__init__(name=name)
        self.hid_dim = hid_dim
        
    def __call__(self, z, s, edge_index, embs):
        embedding = hk.Embed(95, self.hid_dim)
        s_neighbors = embedding(z)

        s_neighbors = hk.LayerNorm(
            axis=-1, 
            create_scale=False, 
            create_offset=False, 
            eps=1e-6  
        )(s_neighbors)

        source, target = edge_index
        messages = embs * s_neighbors[source]

        s_neighbors = jax.ops.segment_sum(
            messages, 
            target,  
            num_segments=s.shape[0]
        )
        
        return s + s_neighbors

class S_vector(hk.Module):
    def __init__(self, hid_dim, name=None):
        super().__init__(name=name)
        self.hid_dim = hid_dim
        
    def __call__(self, s, v, edge_index, emb):
        s = hk.Linear(self.hid_dim)(s)
        s = hk.LayerNorm(axis=-1, create_scale=False, create_offset=False)(s)
        s = jax.nn.silu(s)
        source = edge_index[0].astype(jnp.int32)  
        target = edge_index[1].astype(jnp.int32)  

        node_transform = emb[:, None, :] * v  # (N, 3, H)

        source_s = s[source][:, None, :]  # (E, 1, H)
        
        source_transform = node_transform  # (N, 3, H)
        messages = source_transform * source_s

        E = messages.shape[0]
        messages_flat = messages.reshape(E, 3 * self.hid_dim)
        N = s.shape[0]
        agg_shape = (N, 3 * self.hid_dim)
        aggregated = jnp.zeros(agg_shape).at[target].add(messages_flat)
        return aggregated.reshape(-1, 3, self.hid_dim)

class EquiMessagePassing(hk.Module):
    def __init__(self, 
                 hidden_channels, 
                 num_radial, 
                 head=16, 
                 chi1=32, 
                 chi2=8, 
                 hidden_channels_chi=96,
                 complex_type=jnp.complex64,
                 has_dropout_flag=False,
                 has_norm_before_flag=True,
                 has_norm_after_flag=False,
                 reduce_mode='sum',
                 name=None):
        super().__init__(name=name)
        self.hidden_channels = hidden_channels
        self.num_radial = num_radial
        self.head = head
        self.chi1 = chi1
        self.chi2 = chi2
        self.hidden_channels_chi = hidden_channels_chi
        self.complex_type = complex_type
        self.has_dropout_flag = has_dropout_flag
        self.has_norm_before_flag = has_norm_before_flag
        self.has_norm_after_flag = has_norm_after_flag
        self.reduce_mode = reduce_mode
        
        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(self.hidden_channels)
        self.kernel_transform = hk.Linear(self.chi1)
    def __call__(self, x, vec, edge_index, edge_rbf, weight, edge_vector, edge_mask, rope=None):
        edge_mask = edge_mask.astype(x.dtype)
        edge_mask_2d = edge_mask[:, None]
        edge_mask_3d = edge_mask[:, None, None]
        if rope is not None:
            real, imag = jnp.split(x, 2, axis=-1)
            dy_pre = real + 1j * imag
            dy_pre = dy_pre * rope
            x = jnp.concatenate([jnp.real(dy_pre), jnp.imag(dy_pre)], axis=-1)
            
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        xh = self._build_x_proj()(x)
        
        rbfh = hk.Linear(self.hidden_channels * 3)(edge_rbf) * edge_mask_2d
        weight_proj = self._build_dir_proj()(weight) * edge_mask_2d
        rbfh = rbfh * weight_proj
        
        col, row = edge_index
        messages_x, messages_vec = self.message(
            xh[col], vec[col], rbfh, edge_vector
        )
        messages_x = messages_x * edge_mask_2d
        messages_vec = messages_vec * edge_mask_3d
        
        dx = jax.ops.segment_sum(messages_x, row, x.shape[0])
        dvec = jax.ops.segment_sum(messages_vec, row, vec.shape[0])
        
        if self.has_norm_before_flag:
            dx = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(dx)
            
        dx, dy = dx[..., :self.chi1], dx[..., self.chi1:]
        
        if self.has_norm_after_flag:
            dx = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(dx)
            
        dx = hk.Linear(self.hidden_channels // 2)(dx)
        dx = jnp.cos(dx) + 1j * jnp.sin(dx)
        
        return dx, dy, dvec
    
    def _build_x_proj(self):
        return hk.Sequential([
            hk.Linear(self.hidden_channels),
            jax.nn.silu,
            hk.Linear(self.hidden_channels * 3)
        ])
    
    def _build_dir_proj(self):
        return hk.Sequential([
            hk.Linear(self.hidden_channels * 3),
            jax.nn.silu,
            hk.Linear(self.hidden_channels * 3)
        ])
    
    def _build_diagonal(self):
        return hk.Sequential([
            hk.Linear(self.hidden_channels_chi // 2),
            jax.nn.silu,
            hk.Linear(self.chi2)
        ])
    
    def message(self, xh_j, vec_j, rbfh_ij, r_ij):
        x, xh2, xh3 = jnp.split(xh_j * rbfh_ij, 3, axis=-1)
        xh2 = xh2 * self.inv_sqrt_3
        head_dim = self.hidden_channels_chi // self.head
        
        scale_out = hk.Linear(self.hidden_channels_chi * 2)(x)
        real, imag = jnp.split(scale_out, 2, axis=-1)
        real = real.reshape(x.shape[0], self.head, head_dim)
        imag = imag.reshape(x.shape[0], self.head, head_dim)
        
        phi = real + 1j * imag
        q = phi
        
        a = jnp.ones((q.shape[0], 1, head_dim), dtype=self.complex_type)
        
        kernel_real = hk.get_parameter(
            'kernel_real',
            shape=(self.head + 1, self.hidden_channels_chi // self.head, self.chi2),
            init=hk.initializers.VarianceScaling()
        )
        
        kernel_imag = hk.get_parameter(
            'kernel_imag',
            shape=(self.head + 1, self.hidden_channels_chi // self.head, self.chi2),
            init=hk.initializers.VarianceScaling()
        )
        
        scale_factor = 1 / math.sqrt(self.hidden_channels_chi // self.head)
        kernel_real_part = kernel_real * scale_factor
        kernel_imag_part = kernel_imag * scale_factor
        kernel = kernel_real_part + 1j * kernel_imag_part
        kernel = jnp.broadcast_to(kernel, (q.shape[0],) + kernel.shape)
        
        q_expanded = jnp.concatenate([a, q], axis=1)
        conv = jnp.einsum('ijl,ijlk->ik', q_expanded, kernel)
        
        a_diag = jax.nn.silu(self._build_diagonal()(rbfh_ij))
        
        diachi1 = hk.get_parameter(
            'diachi1',
            shape=(self.chi1,),
            init=hk.initializers.RandomNormal()
        )
        
        b = a_diag[..., jnp.newaxis] * diachi1 + 1.0
        dia = hk.Linear(self.chi1)(b)
        dia_complex = dia + 0j
        
        kernel = jnp.einsum('ik,ikl->il', conv, dia_complex)
        
        kernel_real = self.kernel_transform(jnp.real(kernel))
        kernel_imag = self.kernel_transform(jnp.imag(kernel))
        kernel_angle = jnp.angle(kernel_real + 1j * kernel_imag)
        
        agg = jnp.concatenate([kernel_angle, x], axis=-1)
        
        vec_part1 = vec_j * xh2[:, jnp.newaxis, :]
        r_ij_expanded = r_ij[:, :, jnp.newaxis]
        xh3_expanded = xh3[:, jnp.newaxis, :]
        vec_part2 = xh3_expanded * r_ij_expanded
        vec = vec_part1 + vec_part2
        vec = vec * self.inv_sqrt_h
        
        return agg, vec
class FTE(hk.Module):
    def __init__(self, hidden_channels, name=None):
        super().__init__(name=name)
        self.hidden_channels = hidden_channels

        self.vec_proj = hk.Linear(self.hidden_channels * 2, with_bias=False)
        self.xvec_proj = hk.Sequential([
            hk.Linear(self.hidden_channels),
            jax.nn.silu,
            hk.Linear(self.hidden_channels * 3)
        ])

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_h = 1 / math.sqrt(self.hidden_channels)
    
    def __call__(self, x, vec):

        vec_proj = self.vec_proj(vec)
        vec1, vec2 = jnp.split(vec_proj, 2, axis=-1)
        scalar = jnp.linalg.norm(vec1, axis=1, ord=1)
        vec_dot = jnp.sum(vec1 * vec2, axis=1) * self.inv_sqrt_h

        x_vec_h = self.xvec_proj(jnp.concatenate([x, scalar], axis=-1))
        xvec1, xvec2, xvec3 = jnp.split(x_vec_h, 3, axis=-1)

        dx = (xvec1 + xvec2 + vec_dot) * self.inv_sqrt_2
        dvec = xvec3[:, None] * vec2
        
        return dx, dvec

class AlphaNet_hiku(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.config = config
        if config.dtype == "64":
            self.dtype = jnp.float64
        else:
            self.dtype = jnp.float32
     
        self.z_emb = hk.Embed(95, self.config.hidden_channels)
        self.z_emb_ln = hk.LayerNorm(axis=-1, create_scale=False, create_offset=False)

        self.radial_emb = rbf_emb(
            num_basis=self.config.num_radial, 
            r_max=self.config.cutoff,
            trainable=True
        )
        self.radial_lin = hk.Sequential([
            hk.Linear(self.config.hidden_channels),
            jax.nn.silu,
            hk.Linear(self.config.hidden_channels)
        ])

        self.neighbor_emb = NeighborEmb(self.config.hidden_channels)

        self.s_vector = S_vector(self.config.hidden_channels)

        self.lin = hk.Sequential([
            hk.Linear(self.config.hidden_channels // 4),
            jax.nn.silu,
            hk.Linear(1)
        ])

        self.message_layers = [
            EquiMessagePassing(
                hidden_channels=self.config.hidden_channels,
                num_radial=self.config.num_radial,
                head=self.config.head,
                chi2=self.config.chi2,
                chi1=self.config.mp_chi1,
                has_dropout_flag=self.config.has_dropout_flag,
                has_norm_before_flag=self.config.has_norm_before_flag,
                has_norm_after_flag=self.config.has_norm_after_flag,
                hidden_channels_chi=self.config.hidden_channels_chi,
                complex_type=jnp.complex64 if self.config.dtype == "32" else jnp.complex128,
                reduce_mode=self.config.reduce_mode
            ) for _ in range(self.config.num_layers)
        ]

        self.ftes = [FTE(self.config.hidden_channels) for _ in range(self.config.num_layers)]

        self.last_layer = hk.Linear(self.config.output_dim if self.config.output_dim != 0 else 1)
        self.last_layer_quantum = hk.Linear(1)
        self.zbl = config.zbl
        if self.zbl:
            M = 8
           
            fzbl_w_init = jnp.array([0.187, 0.3769, 0.189, 0.081, 0.003, 0.037, 0.0546, 0.0715], dtype=self.dtype)
          
            fzbl_w_init = jnp.clip(fzbl_w_init, a_min=0.0)
            fzbl_w_init = fzbl_w_init / (jnp.sum(fzbl_w_init) + 1e-12)
            
           
            self.fzbl_w = fzbl_w_init
            self.fzbl_b = jnp.array([3.20, 1.10, 0.102, 0.958, 1.28, 1.14, 1.69, 5], dtype=self.dtype)
            
            self.fzbl_gamma = jnp.array(1.001, dtype=self.dtype)
            self.fzbl_alpha = jnp.array(0.6032, dtype=self.dtype)
            
            
            self.fzbl_E2 = jnp.array(14.399645478425, dtype=self.dtype)  # eV·Å
            self.fzbl_A0 = jnp.array(0.529177210903, dtype=self.dtype)   # Å
     
        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.pi = jnp.pi
        self.eps = jnp.finfo(jnp.float32).eps
    
    def __call__(self, data):
        pos = data.pos.astype(self.dtype)
        batch = data.batch.astype(jnp.int32)
        z = data.z.astype(jnp.int32)
        edge_index = data.edge_index.astype(jnp.int32)
        dist = data.edge_attr.astype(self.dtype)
        vecs = data.edge_vec.astype(self.dtype)
        atom_mask = data.atom_mask
        edge_mask = data.edge_mask
        if atom_mask is None:
            atom_mask = jnp.ones((z.shape[0],), dtype=self.dtype)
        else:
            atom_mask = atom_mask.astype(self.dtype)
        if edge_mask is None:
            edge_mask = jnp.ones((dist.shape[0],), dtype=self.dtype)
        else:
            edge_mask = edge_mask.astype(self.dtype)
        atom_mask_2d = atom_mask[:, None]
        atom_mask_3d = atom_mask[:, None, None]
        edge_mask_2d = edge_mask[:, None]

        z_emb = self.z_emb(z)
        z_emb = self.z_emb_ln(z_emb)
        z_emb = z_emb * atom_mask_2d

        radial_emb = self.radial_emb(dist) * edge_mask_2d
        radial_hidden = self.radial_lin(radial_emb)
        rbounds = 0.5 * (jnp.cos(jnp.clip(dist, 0.0, self.config.cutoff) * jnp.pi / self.config.cutoff) + 1.0)
        radial_hidden = radial_hidden * rbounds[..., None] * edge_mask_2d

        s = self.neighbor_emb(z, z_emb, edge_index, radial_hidden) * atom_mask_2d
        vec = jnp.zeros((s.shape[0], 3, s.shape[1]), dtype=self.dtype)

        j = edge_index[0].astype(jnp.int32)  # source
        i = edge_index[1].astype(jnp.int32)  # target

        edge_diff = jnp.where(edge_mask_2d > 0, vecs / (dist[:, None] + self.config.eps), 0.0)

        ones = edge_mask_2d
        seg_count = jax.ops.segment_sum(ones, i, num_segments=pos.shape[0])
        seg_sum = jax.ops.segment_sum(pos[j] * edge_mask_2d, i, num_segments=pos.shape[0])
        seg_count = jnp.where(seg_count == 0, 1, seg_count)
        mean = seg_sum / seg_count

        edge_cross = jnp.cross(pos[i] - mean[i], pos[j] - mean[i])
        edge_vertical = jnp.cross(edge_diff, edge_cross)
        edge_frame = jnp.stack([edge_diff, edge_cross, edge_vertical], axis=-1)

        edge_diff_expanded = jnp.expand_dims(edge_diff, axis=-1)
        S_i_j = self.s_vector(s, edge_diff_expanded, edge_index, radial_hidden)

        scalrization1 = jnp.sum(
            S_i_j[i][:, :, None, :] * edge_frame[..., None], 
            axis=1
        )
        scalrization2 = jnp.sum(
            S_i_j[j][:, :, None, :] * edge_frame[..., None],
            axis=1
        )

        scalrization1 = scalrization1.at[:, 1, :].set(jnp.abs(scalrization1[:, 1, :]))
        scalrization2 = scalrization2.at[:, 1, :].set(jnp.abs(scalrization2[:, 1, :]))

        scalar3 = self.lin(jnp.transpose(scalrization1, (0, 2, 1)))
        scalar3 += jnp.transpose(scalrization1, (0, 2, 1))[..., 0][:, :, None]
        scalar3 = scalar3.squeeze(-1) / jnp.sqrt(self.config.hidden_channels) 
        
        scalar4 = self.lin(jnp.transpose(scalrization2, (0, 2, 1)))
        scalar4 += jnp.transpose(scalrization2, (0, 2, 1))[..., 0][:, :, None]
        scalar4 = scalar4.squeeze(-1) / jnp.sqrt(self.config.hidden_channels)  
 
        edge_weight = jnp.concatenate([scalar3, scalar4], axis=-1) * rbounds[:, None]
        edge_weight = jnp.concatenate([
            edge_weight, 
            radial_hidden, 
            radial_emb
        ], axis=-1) * edge_mask_2d

        kernel1 = hk.get_parameter(
            'kernel1',
            shape=(self.config.hidden_channels, self.config.main_chi1 * 2),
            init=hk.initializers.VarianceScaling()
        )
        quantum = jnp.einsum('ik,bi->bk', kernel1, z_emb)
        real, imag = jnp.split(quantum, 2, axis=-1)
        quantum = real + 1j * imag 

        kernels_real = hk.get_parameter(
            'kernels_real',
            shape=(self.config.num_layers, self.config.hidden_channels, 
                self.config.main_chi1, self.config.main_chi1),
            init=hk.initializers.VarianceScaling()
        )
        
        kernels_imag = hk.get_parameter(
            'kernels_imag',
            shape=(self.config.num_layers, self.config.hidden_channels, 
                self.config.main_chi1, self.config.main_chi1),
            init=hk.initializers.VarianceScaling()
        )

        a = hk.get_parameter(
            'a',
            shape=(108,),
            init=hk.initializers.Constant(self.config.a)
        )
        
        b = hk.get_parameter(
            'b',
            shape=(108,),
            init=hk.initializers.Constant(self.config.b)
        )

        rope = None
        for idx in range(self.config.num_layers):
            message_layer = self.message_layers[idx]
            fte = self.ftes[idx]
            
            if rope is None:
                rope, ds, dvec = message_layer(s, vec, edge_index, radial_emb, edge_weight, edge_diff, edge_mask, None)
            else:
                rope, ds, dvec = message_layer(s, vec, edge_index, radial_emb, edge_weight, edge_diff, edge_mask, rope)
                
            s = (s + ds) * atom_mask_2d
            vec = (vec + dvec) * atom_mask_3d
 
            kernel_real = kernels_real[idx]
            kernel_imag = kernels_imag[idx]
            kerneli = kernel_real + 1j * kernel_imag
            quantum = jnp.einsum(
                'ikl,bi,bl->bk', 
                kerneli, 
                s + 0j,  
                quantum
            )
            quantum_norm = jnp.abs(quantum)
            quantum = quantum / jnp.where(quantum_norm > self.eps, quantum_norm, 1.0)

            ds, dvec = fte(s, vec)
            s = (s + ds) * atom_mask_2d
            vec = (vec + dvec) * atom_mask_3d

        quantum_real = jnp.real(quantum)
        quantum_imag = jnp.imag(quantum)
        quantum_features = jnp.concatenate([quantum_real, quantum_imag], axis=-1)
        s = self.last_layer(s) + self.last_layer_quantum(quantum_features) / self.config.main_chi1
        a_values = a[z]
        b_values = b[z]
        V_graph = 0
        if self.zbl:
            r_e = dist
            Z_j = z[j]
            Z_i = z[i]

            w = self.fzbl_w  # (M,)
            b = self.fzbl_b  # (M,)
            gamma = self.fzbl_gamma
            alpha = self.fzbl_alpha
            E2 = self.fzbl_E2
            A0 = self.fzbl_A0

            # compute screening length a per edge: a = gamma * 0.8854 * a0 / (Z1^alpha + Z2^alpha)
            denom = jnp.power(Z_j, alpha) + jnp.power(Z_i, alpha)   # (E,)
            denom = jnp.clip(denom, a_min=1e-12)
            a_vals = gamma * 0.8854 * A0 / denom                    # (E,)
            x = r_e / a_vals                                        # (E,)

            # compute phi(x) = sum_i w_i * exp(-b_i * x)  (vectorized)
            # exp(- x[:,None] * b[None,:]) -> (E, M)
            exp_terms = jnp.exp(- x[:, jnp.newaxis] * b[jnp.newaxis, :])   # (E, M)
            phi_vals = exp_terms @ w                                    # (E,)

            # pair potential per edge: V_e = Z1*Z2 * E2 * phi / r
            V_edge = (Z_j * Z_i * E2) * (phi_vals / r_e)              # (E,)
            r_cut = 1.0  # you can make this self.fzbl_rcut buffer if you want configurable value

            # compute taper coefficient: cosine cutoff (smooth)
            # for r in [0, r_cut]: c = 0.5*(cos(pi * r / r_cut) + 1)
            # for r >= r_cut: c = 0
            # for safety, clamp r/r_cut in [0, 1]
            xrc = jnp.clip(r_e / r_cut, 0.0, 1.0)   # (E,)
            # cosine taper
            c = 0.5 * (jnp.cos(jnp.pi * xrc) + 1.0)    # (E,)
            # enforce zero beyond r_cut explicitly (cos already gives 0 at x=1 but clamp keeps numeric safe)
            c = jnp.where(r_e >= r_cut, jnp.zeros_like(c), c)

            # apply taper to edge potential
            V_edge = V_edge * c * edge_mask
            
            # aggregate edge energies to graph-level using jax.ops.segment_sum
            # Note: JAX uses segment_sum instead of scatter_add
            graph_idx = batch[i]  # map receiver node -> graph index (E,)
            V_graph = jax.ops.segment_sum(V_edge, graph_idx, num_segments=1) / 2.0
        if s.ndim == 2:
            s = (a_values[:, None] * s + b_values[:, None]) * atom_mask_2d
        else:
            s = (a_values * s + b_values) * atom_mask
            s = s[:, None]
        
        s = jnp.sum(s) + V_graph
        return jnp.squeeze(s)
        

