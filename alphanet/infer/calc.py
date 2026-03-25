import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from alphanet.models.graph import build_neighbor_topology, graph_from_neighbor_topology
from alphanet.models.model import AlphaNetWrapper

class AlphaNetCalculator(Calculator):
    """
    ASE Calculator for AlphaNet models.

    This calculator wraps an AlphaNet model to perform energy, force, and stress
    calculations within the ASE framework. It automatically handles both periodic
    and non-periodic systems. For non-periodic systems, it creates a large
    supercell with vacuum padding to simulate an isolated molecule.
    """
    implemented_properties = ['energy', 'free_energy', 'forces', 'stress']

    def __init__(
        self,
        ckpt_path,
        config,
        device='cpu',
        precision='32',
        reuse_neighbors=True,
        skin=0.5,
        **kwargs,
    ):
        """
        Initializes the AlphaNetCalculator.

        Args:
            ckpt_path (str): Path to the model checkpoint file (.ckpt or .pt).
            config (object): Model configuration object.
            device (str): Device to run the model on ('cpu' or 'cuda').
            precision (str): Precision for calculations ('32' for float, '64' for double).
            reuse_neighbors (bool): Whether to cache and reuse the neighbor list.
            skin (float): Skin distance (Angstrom) for neighbor list caching.
            **kwargs: Additional arguments for the base ASE Calculator.
        """
        Calculator.__init__(self, **kwargs)

        # --- Model Loading ---
        if precision == "64":
           config.dtype = '64'

        # 获取 LES 配置（若 config 是 All_Config 则有 les 属性）
        les_config = getattr(config, "les", None)
        model_config = getattr(config, "model", config)

        if ckpt_path.endswith('ckpt'):
          self.model = AlphaNetWrapper(model_config, les_config=les_config).to(torch.device(device))
          # Load state dict, ignoring mismatches if any
          self.model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)), strict=False)
        elif ckpt_path.endswith('pt'):
           self.model = torch.load(ckpt_path, map_location=torch.device(device))
        else:
          raise ValueError(f"Unknown checkpoint format for file: {ckpt_path}") 
        
        self.device = torch.device(device)
        self.precision = torch.float32 if precision == "32" else torch.float64
        
        if precision == "64":
          self.model.double()
        
        self.model.eval() # Set model to evaluation mode
        self.model.to(self.device)
        self.config = config
        self.reuse_neighbors = reuse_neighbors
        self.supports_neighbor_cache = hasattr(self.model, "forward_graph")
        self.skin = max(float(skin), 0.0)
        self._neighbor_topology = None
        self._reference_positions = None
        self._reference_cell = None
        self._reference_numbers = None
        self._reference_pbc = None
        self._neighbor_cache_stats = {"rebuilds": 0, "reuses": 0}

    @property
    def neighbor_cache_stats(self):
        return dict(self._neighbor_cache_stats)

    def reset_neighbor_cache(self):
        self._neighbor_topology = None
        self._reference_positions = None
        self._reference_cell = None
        self._reference_numbers = None
        self._reference_pbc = None

    def _prepare_atoms(self):
        if not self.atoms.pbc.any():
            print("Non-periodic system detected. Automatically adding a large vacuum box for calculation.")
            calc_atoms = self.atoms.copy()
            padding = 20.0
            new_cell_dims = calc_atoms.get_positions().ptp(axis=0) + padding
            calc_atoms.set_cell(np.diag(new_cell_dims))
            calc_atoms.center()
            calc_atoms.pbc = True
            return calc_atoms
        return self.atoms

    def _max_displacement_since_rebuild(self, positions, cell, pbc):
        if self._reference_positions is None or self._reference_cell is None:
            return float("inf")
        aligned_positions = self._align_positions_to_reference(positions, cell, pbc)
        delta_cart = aligned_positions - self._reference_positions
        distances = np.linalg.norm(delta_cart, axis=1)
        return float(np.max(distances)) if distances.size else 0.0

    def _align_positions_to_reference(self, positions, cell, pbc):
        if (
            self._reference_positions is None
            or self._reference_cell is None
            or not np.allclose(cell, self._reference_cell, atol=1e-12, rtol=0.0)
        ):
            return positions.copy()
        inverse_cell = np.linalg.inv(cell)
        current_frac = positions @ inverse_cell
        reference_frac = self._reference_positions @ inverse_cell
        delta_frac = current_frac - reference_frac
        periodic_axes = np.asarray(pbc, dtype=bool)
        delta_frac[:, periodic_axes] -= np.round(delta_frac[:, periodic_axes])
        return (reference_frac + delta_frac) @ cell

    def _should_rebuild_topology(self, positions, cell, numbers, pbc):
        if not self.reuse_neighbors or self.skin <= 0.0:
            return True
        if self._neighbor_topology is None:
            return True
        if self._reference_numbers is None or numbers.shape != self._reference_numbers.shape:
            return True
        if not np.array_equal(numbers, self._reference_numbers):
            return True
        if self._reference_pbc is None or not np.array_equal(np.asarray(pbc, dtype=bool), self._reference_pbc):
            return True
        if self._reference_cell is None or not np.allclose(cell, self._reference_cell, atol=1e-12, rtol=0.0):
            return True
        return self._max_displacement_since_rebuild(positions, cell, pbc) > (0.5 * self.skin)

    def _build_or_reuse_topology(self, positions, cell_array, numbers, pbc, pos, natoms):
        if self._should_rebuild_topology(positions, cell_array, numbers, pbc):
            self._neighbor_topology = build_neighbor_topology(
                pos=pos.detach(),
                natoms=natoms,
                cell=torch.tensor(cell_array, dtype=self.precision, device=self.device).detach(),
                cutoff=self.config.cutoff,
                skin=self.skin,
                precision=self.precision,
                numbers=numbers,
            )
            self._reference_positions = positions.copy()
            self._reference_cell = cell_array.copy()
            self._reference_numbers = numbers.copy()
            self._reference_pbc = np.asarray(pbc, dtype=bool).copy()
            self._neighbor_cache_stats["rebuilds"] += 1
        else:
            self._neighbor_cache_stats["reuses"] += 1
        return self._neighbor_topology

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Performs the calculation of energy, forces, and stress.

        Args:
            atoms (ase.Atoms): The atoms object to calculate properties for.
            properties (list of str): List of properties to calculate.
            system_changes (list of str): List of changes since the last calculation.
        """
        Calculator.calculate(self, atoms, properties, system_changes)
        properties = properties or ['energy']
        calc_atoms = self._prepare_atoms()
        needs_stress = 'stress' in properties
        needs_forces = needs_stress or ('forces' in properties)
        grad_enabled = needs_forces or needs_stress

        # --- Prepare Tensors for the Model ---
        atomic_numbers = calc_atoms.get_atomic_numbers()
        wrapped_positions = calc_atoms.get_positions(wrap=True)
        cell_array = np.array(calc_atoms.get_cell(complete=True))
        z = torch.tensor(atomic_numbers, dtype=torch.long, device=self.device)
        natoms = torch.tensor(
            [len(calc_atoms)], 
            dtype=torch.int64, 
            device=self.device
        )
        batch = torch.zeros_like(z).to(self.device)

        # --- Run Model Inference ---
        use_neighbor_cache = (
            self.reuse_neighbors
            and self.supports_neighbor_cache
            and calc_atoms.pbc.any()
            and not needs_stress
        )
        positions_for_model = wrapped_positions
        if use_neighbor_cache:
            positions_for_model = self._align_positions_to_reference(
                wrapped_positions,
                cell_array,
                calc_atoms.pbc,
            )
        pos = torch.tensor(
            positions_for_model,
            dtype=self.precision,
            device=self.device,
            requires_grad=grad_enabled,
        )
        cell = torch.tensor(
            cell_array,
            dtype=self.precision,
            device=self.device
        ) if calc_atoms.pbc.any() else None
        with torch.set_grad_enabled(grad_enabled):
            if use_neighbor_cache:
                topology = self._build_or_reuse_topology(
                    positions=positions_for_model,
                    cell_array=cell_array,
                    numbers=atomic_numbers,
                    pbc=calc_atoms.pbc,
                    pos=pos,
                    natoms=natoms,
                )
                graph_data = graph_from_neighbor_topology(
                    pos=pos,
                    z=z,
                    natoms=natoms,
                    batch=batch,
                    topology=topology,
                    cell=cell,
                    cutoff=self.config.cutoff,
                    dtype=self.precision,
                )
                energy, forces, stress = self.model.forward_graph(
                    graph_data,
                    prefix="infer",
                    compute_forces=needs_forces,
                    compute_stress=False,
                )
            else:
                energy, forces, stress = self.model(
                    pos,
                    z,
                    batch,
                    natoms,
                    cell,
                    "infer",
                    compute_forces=needs_forces,
                    compute_stress=needs_stress,
                )
        
        # --- Store Results ---
        self.results['energy'] = energy.detach().cpu().item()
        self.results['free_energy'] = self.results['energy']  
        
        if forces is not None:
            self.results['forces'] = forces.detach().cpu().numpy()
        
        if stress is not None:
            # Convert the model's 3x3 stress tensor to ASE's Voigt notation (6-element vector)
            stress_matrix = stress.detach().cpu().numpy()
            self.results['stress'] = np.array([
                stress_matrix[0, 0],  # xx
                stress_matrix[1, 1],  # yy
                stress_matrix[2, 2],  # zz
                stress_matrix[1, 2],  # yz
                stress_matrix[0, 2],  # xz
                stress_matrix[0, 1]   # xy
            ])
