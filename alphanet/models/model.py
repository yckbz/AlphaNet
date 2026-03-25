import torch
import time
from torch import Tensor
from typing import Optional
from alphanet.models.alphanet import AlphaNet
from alphanet.models.graph import GraphData, process_positions_and_edges
from alphanet.config import AlphaConfig


class AlphaNetWrapper(torch.nn.Module):
    def __init__(
        self,
        config: AlphaConfig,
        les_config=None,
    ):
        super(AlphaNetWrapper, self).__init__()

        # 根据配置选择实例化 AlphaNet 或 AlphaNetLES
        use_les = getattr(config, "use_les", False) or (
            les_config is not None and getattr(les_config, "use_les", False)
        )

        if use_les:
            from alphanet.models.alphanet_les import AlphaNetLES
            from alphanet.config import LESConfig

            if les_config is None:
                les_config = LESConfig()
            self.model = AlphaNetLES(config, les_config)
        else:
            self.model = AlphaNet(config)

        self.cutoff = config.cutoff
        self.compute_forces = config.compute_forces
        self.compute_stress = config.compute_stress
        self.use_pbc = config.use_pbc
        self.precision =  torch.float32 if config.dtype == "32" else torch.float64

    def _resolve_compute_flags(
        self,
        compute_forces: Optional[bool],
        compute_stress: Optional[bool],
    ):
        resolved_forces = self.compute_forces if compute_forces is None else compute_forces
        resolved_stress = self.compute_stress if compute_stress is None else compute_stress
        return resolved_forces, resolved_stress

    def forward(
            self,
            pos: Tensor,
            z: Tensor,
            batch: Tensor,
            natoms: Tensor,
            cell: Optional[Tensor] = None,
            prefix: str = 'infer',
            compute_forces: Optional[bool] = None,
            compute_stress: Optional[bool] = None,
            return_atom_energy: bool = False):
        compute_forces, compute_stress = self._resolve_compute_flags(
            compute_forces,
            compute_stress,
        )
        processed_data = process_positions_and_edges(
            pos=pos,
            z=z,
            natoms=natoms,
            batch=batch,
            cell=cell,
            compute_forces=compute_forces,
            compute_stress=compute_stress,
            use_pbc=self.use_pbc,
            cutoff=self.cutoff,
            dtype=self.precision
        )
        return self.forward_graph(
            processed_data,
            prefix=prefix,
            compute_forces=compute_forces,
            compute_stress=compute_stress,
            return_atom_energy=return_atom_energy,
        )

    def forward_graph(
        self,
        graph_data: GraphData,
        prefix: str = "infer",
        compute_forces: Optional[bool] = None,
        compute_stress: Optional[bool] = None,
        return_atom_energy: bool = False,
    ):
        compute_forces, compute_stress = self._resolve_compute_flags(
            compute_forces,
            compute_stress,
        )
        return self.model(
            graph_data,
            prefix,
            compute_forces=compute_forces,
            compute_stress=compute_stress,
            return_atom_energy=return_atom_energy,
        )
