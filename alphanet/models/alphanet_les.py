import math
import pathlib
import sys
from math import pi
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import Embedding

from alphanet.models.alphanet import (
    EquiMessagePassing,
    FTE,
    NeighborEmb,
    S_vector,
    rbf_emb,
    scatter,
)
from alphanet.models.graph import GraphData

# Make the vendored LES library (third_party/) importable as a fallback.
# If `les` is already installed via pip, that version takes priority (append).
_third_party = pathlib.Path(__file__).resolve().parents[2] / "third_party"
if _third_party.is_dir() and str(_third_party) not in sys.path:
    sys.path.append(str(_third_party))

from les import Les


class LESReadout(nn.Module):
    """MLP head that maps AlphaNet atom features (s + quantum) to latent charges.

    Architecture: LayerNorm -> [Linear -> SiLU -> LayerNorm] * len(hidden_layers) -> Linear

    Args:
        input_dim: Input feature dimension. Either hidden_channels (s only) or
                   hidden_channels + chi1*2 (s concatenated with quantum real/imag parts).
        n_latent_charges: Number of latent charge channels per atom (n_q). Defaults to 1.
        hidden_layers: List of hidden layer widths, e.g. [64, 32].
        output_scaling_factor: Linear scale applied to the output. A small value (e.g. 0.1)
                               keeps the LES contribution small at initialization for stable training.
    """

    def __init__(
        self,
        input_dim: int,
        n_latent_charges: int = 1,
        hidden_layers: list = None,
        output_scaling_factor: float = 0.1,
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [64, 32]
        self.output_scaling_factor = output_scaling_factor

        self.input_norm = nn.LayerNorm(input_dim)

        # Build MLP: each hidden layer is followed by SiLU activation and LayerNorm.
        layers: List[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.SiLU())
            layers.append(nn.LayerNorm(h_dim))
            in_dim = h_dim
        # Final linear projection, no activation.
        layers.append(nn.Linear(in_dim, n_latent_charges))

        self.mlp = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: [n_atoms, input_dim]
        Returns:
            latent_charges: [n_atoms, n_q]
        """
        x = self.input_norm(features)
        charges = self.mlp(x)
        return charges * self.output_scaling_factor


class AlphaNetLES(nn.Module):
    """AlphaNet with LES long-range electrostatics.

    Total energy:
        E_total = E_short (AlphaNet) + E_lr (LES Ewald summation)

    The two readout paths share the same message-passing representation.
    Training is fully end-to-end: latent charges are inferred from energy/force
    labels without any explicit charge supervision.

    Args:
        config: AlphaConfig — backbone network hyperparameters.
        les_config: LESConfig — LES module hyperparameters.
        device: Torch device. Inferred from config.device when None.

    Notes:
        - All backbone parameters are identical to vanilla AlphaNet, so a pretrained
          AlphaNet checkpoint can be loaded with strict=False. Missing keys will be
          les_readout.* and les.*, which are randomly initialized.
        - Stress correctness: get_symmetric_displacement modifies both positions and cell
          so that they depend on the displacement variable. E_lr depends on positions
          (via k·r terms) and cell (via volume and reciprocal lattice vectors), so
          dE_lr/d(displacement) is correctly captured by autograd.
    """

    def __init__(
        self,
        config,
        les_config,
        device=None,
    ):
        super(AlphaNetLES, self).__init__()

        # Resolve device: prefer config.device to match vanilla AlphaNet behaviour.
        if device is None:
            cfg_device = getattr(config, "device", "cpu")
            device = torch.device(cfg_device)
        self.device = device
        self.complex_type = torch.complex64 if config.dtype == "32" else torch.complex128
        self.eps = 1e-9
        self.num_layers = config.num_layers
        self.hidden_channels = config.hidden_channels
        self.a = nn.Parameter(torch.ones(108) * config.a)
        self.b = nn.Parameter(torch.ones(108) * config.b)
        self.cutoff = config.cutoff
        self.readout = config.readout
        self.chi1 = config.main_chi1

        self.use_sigmoid = config.use_sigmoid
        self.num_targets = config.output_dim if config.output_dim != 0 else 1
        self.compute_forces = config.compute_forces
        self.compute_stress = config.compute_stress

        # ── AlphaNet backbone components (identical to vanilla AlphaNet) ──────────
        self.z_emb_ln = nn.LayerNorm(config.hidden_channels, elementwise_affine=False)
        self.z_emb = Embedding(95, config.hidden_channels)
        self.kernel1 = nn.Parameter(
            torch.randn((config.hidden_channels, self.chi1 * 2), device=self.device)
        )
        self.radial_emb = rbf_emb(config.num_radial, config.cutoff)
        self.radial_lin = nn.Sequential(
            nn.Linear(config.num_radial, config.hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(config.hidden_channels, config.hidden_channels),
        )
        self.pi = pi
        self.neighbor_emb = NeighborEmb(config.hidden_channels)
        self.S_vector = S_vector(config.hidden_channels)
        self.lin = nn.Sequential(
            nn.Linear(3, config.hidden_channels // 4),
            nn.SiLU(inplace=True),
            nn.Linear(config.hidden_channels // 4, 1),
        )

        self.message_layers = nn.ModuleList()
        self.FTEs = nn.ModuleList()
        self.zbl = config.zbl

        if self.zbl:
            self.register_buffer(
                "fzbl_w",
                torch.tensor(
                    [0.187, 0.3769, 0.189, 0.081, 0.003, 0.037, 0.0546, 0.0715],
                    dtype=torch.get_default_dtype(),
                ),
            )
            self.register_buffer(
                "fzbl_b",
                torch.tensor(
                    [3.20, 1.10, 0.102, 0.958, 1.28, 1.14, 1.69, 5],
                    dtype=torch.get_default_dtype(),
                ),
            )
            with torch.no_grad():
                w = getattr(self, "fzbl_w")
                w = w.clamp(min=0.0)
                w = w / (w.sum() + 1e-12)
                self.fzbl_w.copy_(w)
            self.register_buffer(
                "fzbl_gamma", torch.tensor(1.001, dtype=torch.get_default_dtype())
            )
            self.register_buffer(
                "fzbl_alpha", torch.tensor(0.6032, dtype=torch.get_default_dtype())
            )
            self.register_buffer(
                "fzbl_E2",
                torch.tensor(14.399645478425, dtype=torch.get_default_dtype()),
            )
            self.register_buffer(
                "fzbl_A0",
                torch.tensor(0.529177210903, dtype=torch.get_default_dtype()),
            )

        kernels_real_list = []
        kernels_imag_list = []
        for _ in range(config.num_layers):
            self.message_layers.append(
                EquiMessagePassing(
                    hidden_channels=config.hidden_channels,
                    num_radial=config.num_radial,
                    head=config.head,
                    chi2=config.chi2,
                    chi1=config.mp_chi1,
                    has_dropout_flag=config.has_dropout_flag,
                    has_norm_before_flag=config.has_norm_before_flag,
                    has_norm_after_flag=config.has_norm_after_flag,
                    hidden_channels_chi=config.hidden_channels_chi,
                    complex_type=self.complex_type,
                    device=device,
                    reduce_mode=config.reduce_mode,
                )
            )
            self.FTEs.append(FTE(config.hidden_channels))
            kernels_real_list.append(
                torch.randn((config.hidden_channels, self.chi1, self.chi1))
            )
            kernels_imag_list.append(
                torch.randn((config.hidden_channels, self.chi1, self.chi1))
            )

        self.kernels_real = nn.Parameter(torch.stack(kernels_real_list))
        self.kernels_imag = nn.Parameter(torch.stack(kernels_imag_list))

        # Short-range readout (identical to vanilla AlphaNet).
        self.last_layer = nn.Linear(config.hidden_channels, self.num_targets)
        self.last_layer_quantum = nn.Linear(self.chi1 * 2, self.num_targets)

        self.use_quantum_for_charges = les_config.use_quantum_for_charges

        # LES readout input dim: s alone, or s concatenated with quantum real/imag.
        if self.use_quantum_for_charges:
            les_readout_input_dim = config.hidden_channels + self.chi1 * 2
        else:
            les_readout_input_dim = config.hidden_channels

        self.les_readout = LESReadout(
            input_dim=les_readout_input_dim,
            n_latent_charges=les_config.n_latent_charges,
            hidden_layers=list(les_config.les_readout_hidden),
            output_scaling_factor=les_config.output_scaling_factor,
        )

        # LES Ewald module: use_atomwise=False because latent charges come from les_readout.
        les_arguments = {
            "sigma": les_config.sigma,
            "dl": les_config.dl,
            "remove_self_interaction": les_config.remove_self_interaction,
            "use_atomwise": False,
        }
        self.les = Les(les_arguments)

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.reset_parameters()

    def reset_parameters(self):
        self.z_emb.reset_parameters()
        for layer in self.message_layers:
            layer.reset_parameters()
        for layer in self.FTEs:
            layer.reset_parameters()
        self.last_layer.reset_parameters()
        for layer in self.radial_lin:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for layer in self.lin:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(
        self,
        data: GraphData,
        prefix: str,
        compute_forces: Optional[bool] = None,
        compute_stress: Optional[bool] = None,
        return_atom_energy: bool = False,
    ):
        """Forward pass.

        Args:
            data: GraphData with pos, z, batch, edge_index, edge_vec, cell, displacement.
            prefix: 'train' enables create_graph=True for higher-order gradients; 'infer' disables it.
            compute_forces: Override instance-level flag if provided.
            compute_stress: Override instance-level flag if provided.
            return_atom_energy: If True, also return per-atom short-range energies.

        Returns:
            Tuple (total_energy, forces, stress), same format as vanilla AlphaNet.
        """
        compute_forces = self.compute_forces if compute_forces is None else compute_forces
        compute_stress = self.compute_stress if compute_stress is None else compute_stress

        pos = data.pos
        batch = data.batch
        z = data.z.long()
        edge_index = data.edge_index
        vecs = data.edge_vec

        # ── Pre-message-passing feature preparation (identical to vanilla AlphaNet) ──
        dist = torch.linalg.norm(vecs, dim=1)
        z_emb = self.z_emb_ln(self.z_emb(z))
        radial_emb = self.radial_emb(dist)
        radial_hidden = self.radial_lin(radial_emb)
        rbounds = 0.5 * (torch.cos(dist * self.pi / self.cutoff) + 1.0)
        radial_hidden = rbounds.unsqueeze(-1) * radial_hidden

        s = self.neighbor_emb(z, z_emb, edge_index, radial_hidden)
        vec = torch.zeros(s.size(0), 3, s.size(1), device=s.device)

        j = edge_index[0]
        i = edge_index[1]
        edge_diff = vecs / (dist.unsqueeze(1) + self.eps)

        edge_vec_mean = scatter(vecs, i, reduce="mean", dim=0)
        edge_cross = torch.linalg.cross(vecs, edge_vec_mean[i])
        edge_vertical = torch.linalg.cross(edge_diff, edge_cross)
        edge_frame = torch.cat(
            (
                edge_diff.unsqueeze(-1),
                edge_cross.unsqueeze(-1),
                edge_vertical.unsqueeze(-1),
            ),
            dim=-1,
        )

        S_i_j = self.S_vector(s, edge_diff.unsqueeze(-1), edge_index, radial_hidden)

        scalrization1 = torch.sum(S_i_j[i].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1)
        scalrization2 = torch.sum(S_i_j[j].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1)
        scalrization1[:, 1, :] = torch.square(scalrization1[:, 1, :].clone())
        scalrization2[:, 1, :] = torch.square(scalrization2[:, 1, :].clone())

        scalar3 = (
            self.lin(torch.permute(scalrization1, (0, 2, 1)))
            + torch.permute(scalrization1, (0, 2, 1))[:, :, 0].unsqueeze(2)
        ).squeeze(-1) / math.sqrt(self.hidden_channels)
        scalar4 = (
            self.lin(torch.permute(scalrization2, (0, 2, 1)))
            + torch.permute(scalrization2, (0, 2, 1))[:, :, 0].unsqueeze(2)
        ).squeeze(-1) / math.sqrt(self.hidden_channels)

        edge_weight = torch.cat((scalar3, scalar4), dim=-1) * rbounds.unsqueeze(-1)
        edge_weight = torch.cat((edge_weight, radial_hidden, radial_emb), dim=-1)

        # Quantum state initialization.
        quantum = torch.einsum("ik,bi->bk", self.kernel1, z_emb)
        real, imagine = torch.split(quantum, self.chi1, dim=-1)
        quantum = torch.complex(real, imagine)

        # ── Message-passing loop (identical to vanilla AlphaNet) ──────────────────
        rope = None
        for id, (message_layer, fte) in enumerate(zip(self.message_layers, self.FTEs)):
            rope, ds, dvec = message_layer(
                s, vec, edge_index, radial_emb, edge_weight, edge_diff, rope
            )
            s = s + ds
            vec = vec + dvec

            kerneli = torch.complex(self.kernels_real[id], self.kernels_imag[id])
            quantum = torch.einsum("ikl,bi,bl->bk", kerneli, s.to(self.complex_type), quantum)
            quantum = quantum / (self.eps + quantum.abs().to(self.complex_type))

            ds, dvec = fte(s, vec)
            s = s + ds
            vec = vec + dvec

        # ── Dual readout: short-range and long-range branches ─────────────────────

        # Branch 1: short-range energy (identical to vanilla AlphaNet).
        s_energy = (
            self.last_layer(s)
            + self.last_layer_quantum(torch.cat([quantum.real, quantum.imag], dim=-1))
            / self.chi1
        )

        if s_energy.dim() == 2:
            s_energy = self.a[z].unsqueeze(1) * s_energy + self.b[z].unsqueeze(1)
        elif s_energy.dim() == 1:
            s_energy = (self.a[z] * s_energy + self.b[z]).unsqueeze(1)
        else:
            raise ValueError(f"Unexpected shape of s_energy: {s_energy.shape}")

        atom_energy_short = (
            s_energy.squeeze(-1) if s_energy.dim() == 2 and s_energy.size(-1) == 1 else s_energy
        )
        E_short = scatter(atom_energy_short, batch, dim=0, reduce=self.readout).squeeze()

        # Branch 2: long-range energy via LES Ewald summation.
        # Concatenate s with quantum real/imag parts if use_quantum_for_charges is True.
        if self.use_quantum_for_charges:
            les_features = torch.cat([s, quantum.real, quantum.imag], dim=-1)
        else:
            les_features = s

        latent_charges = self.les_readout(les_features)  # [n_atoms, n_q]

        # Ensure cell shape is [batch_size, 3, 3] as expected by LES.
        # GraphData.cell is already [batch_size, 3, 3] after check_and_reshape_cell,
        # and retains the gradient graph from get_symmetric_displacement (stress).
        cell_for_les = data.cell
        if cell_for_les is None:
            # Non-periodic: pass zero cell; LES detects det(cell)<1e-6 and uses real-space sum.
            num_graphs = int(torch.max(batch).item()) + 1
            cell_for_les = torch.zeros(
                (num_graphs, 3, 3), dtype=pos.dtype, device=pos.device
            )
        elif cell_for_les.dim() == 2:
            # Defensive reshape in case cell arrives as [batch*3, 3].
            num_graphs = int(torch.max(batch).item()) + 1
            cell_for_les = cell_for_les.view(num_graphs, 3, 3)

        les_output = self.les(
            positions=pos,
            cell=cell_for_les,
            latent_charges=latent_charges,
            batch=batch,
            compute_energy=True,
            compute_bec=False,
        )
        E_lr = les_output["E_lr"]  # [batch_size]

        total_energy = E_short + E_lr

        if self.use_sigmoid:
            if return_atom_energy:
                raise ValueError(
                    "Per-atom energy is not defined when sigmoid readout is enabled"
                )
            total_energy = torch.sigmoid((total_energy - 0.5) * 5)

        # ── Forces and stress: autograd on E_total flows through both branches ─────
        if compute_forces and compute_stress:
            if data.displacement is not None:
                stress, forces = self.cal_stress_and_force(
                    total_energy, pos, data.displacement, data.cell, prefix
                )
                stress = stress.view(-1, 3)
            else:
                stress = None
                forces = None
            if return_atom_energy:
                return total_energy, forces, stress, atom_energy_short
            return total_energy, forces, stress
        elif compute_forces:
            forces = self.cal_forces(total_energy, pos, prefix)
            if return_atom_energy:
                return total_energy, forces, None, atom_energy_short
            return total_energy, forces, None

        if return_atom_energy:
            return total_energy, None, None, atom_energy_short
        return total_energy, None, None

    def cal_forces(self, energy: Tensor, positions: Tensor, prefix: str = "infer") -> Tensor:
        graph = prefix == "train"
        grad_outputs = torch.jit.annotate(
            List[Optional[Tensor]], [torch.ones_like(energy)]
        )
        forces = torch.autograd.grad(
            outputs=[energy],
            inputs=[positions],
            grad_outputs=grad_outputs,
            create_graph=graph,
            retain_graph=graph,
            allow_unused=True,
        )[0]
        assert forces is not None, "Gradient should not be None"
        return -forces

    def cal_stress_and_force(
        self,
        energy: Tensor,
        positions: Tensor,
        displacement: Optional[Tensor],
        cell: Tensor,
        prefix: str,
    ) -> Tuple[Tensor, Tensor]:
        if displacement is None:
            raise ValueError("displacement cannot be None for stress calculation")
        graph = prefix == "train"
        grad_outputs = torch.jit.annotate(
            List[Optional[Tensor]], [torch.ones_like(energy)]
        )
        output = torch.autograd.grad(
            [energy],
            [displacement, positions],
            grad_outputs=grad_outputs,
            create_graph=graph,
            retain_graph=graph,
            allow_unused=True,
        )
        virial = (
            output[0]
            if output[0] is not None
            else torch.zeros((3, 3), device=cell.device)
        )
        assert virial is not None, "Virial tensor should not be None"
        volume = torch.abs(torch.linalg.det(cell))
        volume_expanded = volume.reshape(-1, 1, 1)
        stress = virial / volume_expanded
        force = output[1]
        assert force is not None, "Forces tensor should not be None"
        return stress, -force
