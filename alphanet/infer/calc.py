import torch
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.data import atomic_numbers
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

    def __init__(self, ckpt_path, config, device='cpu', precision='32', **kwargs):
        """
        Initializes the AlphaNetCalculator.

        Args:
            ckpt_path (str): Path to the model checkpoint file (.ckpt or .pt).
            config (object): Model configuration object.
            device (str): Device to run the model on ('cpu' or 'cuda').
            precision (str): Precision for calculations ('32' for float, '64' for double).
            **kwargs: Additional arguments for the base ASE Calculator.
        """
        Calculator.__init__(self, **kwargs)
        
        # --- Model Loading ---
        if precision == "64":
           config.dtype = '64'
        if ckpt_path.endswith('ckpt'):
          self.model = AlphaNetWrapper(config).to(torch.device(device))
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
        
        # --- Handle Non-Periodic Systems ---
        # If the system is not periodic (e.g., a molecule), we create a copy and
        # place it in a large box with vacuum padding. This allows the model,
        # which assumes periodicity, to treat it as an isolated system.
        if not self.atoms.pbc.any():
            print("Non-periodic system detected. Automatically adding a large vacuum box for calculation.")
            calc_atoms = self.atoms.copy()
            # Add 20 Å of vacuum padding around the molecule
            padding = 20.0
            new_cell_dims = calc_atoms.get_positions().ptp(axis=0) + padding
            calc_atoms.set_cell(np.diag(new_cell_dims))
            calc_atoms.center()
            calc_atoms.pbc = True # Treat it as periodic now
        else:
            calc_atoms = self.atoms

        # --- Prepare Tensors for the Model ---
#        z = torch.tensor(
#            [atomic_numbers[atom.symbol] for atom in calc_atoms], 
#            dtype=torch.long, 
#            device=self.device
#        )
        z = torch.tensor(
            calc_atoms.get_atomic_numbers(),
            dtype=torch.long,
            device=self.device
        )
        pos = torch.tensor(
            calc_atoms.get_positions(wrap=True), 
            dtype=self.precision, 
            device=self.device, 
            requires_grad=(self.config.compute_forces)  
        )
       
        # Cell should only be provided if the system is periodic
#        cell = torch.tensor(
#            calc_atoms.get_cell(complete=True), 
#            dtype=self.precision, 
#            device=self.device
#        ) if calc_atoms.pbc.any() else None
        
        cell = torch.tensor(
            np.array(calc_atoms.get_cell(complete=True)),
            dtype=self.precision,
            device=self.device
        ) if calc_atoms.pbc.any() else None

        natoms = torch.tensor(
            [len(calc_atoms)], 
            dtype=torch.int64, 
            device=self.device
        )
        batch = torch.zeros_like(z).to(self.device)

        # --- Run Model Inference ---
        with torch.set_grad_enabled(self.config.compute_forces):
            energy, forces, stress = self.model(pos, z, batch, natoms, cell, "infer")
        
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


