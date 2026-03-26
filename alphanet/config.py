"""Pydantic model for default configuration and validation."""

import subprocess
import json
#import torch
from typing import Literal, Dict, Optional
from pydantic_settings import BaseSettings

try:
    VERSION = (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    )
except Exception:
    VERSION = "NA"
    pass



class TrainConfig(BaseSettings):
    """Hyperparameter schema for training."""
    epochs: int = 1000
    batch_size: int = 32
    vt_batch_size: int = 32
    lr: float = 0.0005
    optimizer: str = "radam"
    lr_decay_factor: float = 0.5
    lr_decay_step_size: int = 150
    weight_decay: float = 0
    save_dir: str = ""
    log_dir: str = ""
    num_workers: int = 0
    accumulation_steps: int = 1
    disable_tqdm: bool = False
    scheduler: str = "steplr" #I prefer Consineanealing
    norm_label: bool = False
    device: str = "cuda"
    energy_loss: str = "mae"  # My experiments are basically using MAE loss, I think MSE would also work but you may need to adjust the weight of the loss.
    force_loss: str = "mae"  
    stress_loss: str = "mae"
    energy_metric: str = "mae"
    force_metric: str = "mae"  
    stress_metric: str = "mae"
    energy_coef: float = 1.0  #Usually, I would set the weight of the losses energy: focre: stress: 4:100:100 for systems that are not too large(<300 atoms). If the systems are large or the energy per atom has large value, I would try a dynamic strategy for now, for example, first, train it with 0.01:100:100 with lr 5e-4 and then gradually rise to 1:100:100 the weight of energy loss and decrease the lr to 1e-5. That may not be very conveinient and we will try to make it done systematically.
    force_coef: float = 0.0  
    stress_coef: float = 0.0
    eval_steps: int = 1  

class DataConfig(BaseSettings):
    """Hyperparameter schema for dataset."""
    root: str = "dataset/"
    dataset_name: str = "qm9"
    target: str = "U0"
    train_size: Optional[int] = None
    valid_size: Optional[int] = None
    test_size: Optional[int] = None
    train_dataset: Optional[str] = None
    valid_dataset: Optional[str] = None
    test_dataset: Optional[str] = None
    seed: int = 42

class AlphaConfig(BaseSettings):
    """Hyperparameter schema for AlphaNet. The main keywords you need to adjust maybe num_layers, hidden_channels, cutoff, head"""

    name: str =  "Alphanet"
    num_layers: int = 3
    num_targets: int = 1
    output_dim: int = 1
    readout: str = "sum"
    use_pbc: bool = True
    compute_forces: bool = False
    compute_stress: bool = False
    eps: float = 1e-10
    hidden_channels: int = 128
    cutoff: float = 5.0
    num_radial: int = 96
    dtype: str = "32"  # datatype 32 or 64
    use_sigmoid: bool = False
    head: int = 16
    a: float = 1
    b: float = 1
    main_chi1: int = 24
    mp_chi1: int = 24
    chi2: int = 6
    hidden_channels_chi: int = 96
    has_dropout_flag: bool = True
    has_norm_before_flag: bool = True
    has_norm_after_flag: bool = False
    reduce_mode: str = "sum"
    zbl: bool = False
    use_les: bool = False
    zbl_w: Optional[list] = [0.187,0.3769,0.189,0.081,0.003,0.037,0.0546,0.0715]
    zbl_b: Optional[list] = [3.20,1.10,0.102,0.958,1.28,1.14,1.69,5]
    zbl_gamma: float = 1.001
    zbl_alpha: float = 0.6032
    zbl_E2: float = 14.399645478425
    zbl_A0: float = 0.529177210903
    device: str = "cuda"

    

        
class LESConfig(BaseSettings):
    """Configuration for the LES (Latent Ewald Summation) long-range interaction module.

    Pass as a top-level "les" key in the JSON config file, e.g.:
        { "les": { "use_les": true, "sigma": 1.0, "dl": 2.0 } }
    """
    use_les: bool = False
    n_latent_charges: int = 1              # Number of latent charge channels per atom (n_q).
    les_readout_hidden: list = [64, 32]    # Hidden layer widths of the LES readout MLP.
    use_quantum_for_charges: bool = True   # Use both s and quantum state to predict latent charges.
    output_scaling_factor: float = 0.1    # Output scale for latent charges; keeps LES contribution small at init.
    # Ewald summation parameters
    sigma: float = 1.0                     # Gaussian width (Å) controlling real/reciprocal space splitting.
    dl: float = 2.0                        # Reciprocal-space grid spacing (Å); smaller = more k-points = more accurate.
    remove_self_interaction: bool = True   # Remove self-interaction term (recommended).


class All_Config:
    def __init__(self, data=None, model=None, train=None, les=None):

        self.data = DataConfig(**data) if data else DataConfig()
        self.model = AlphaConfig(**model) if model else AlphaConfig()
        self.train = TrainConfig(**train) if train else TrainConfig()
        self.les = LESConfig(**les) if les else LESConfig()

    def __getattr__(self, name):

        if hasattr(self.train, name):
            return getattr(self.train, name)
        elif hasattr(self.data, name):
            return getattr(self.data, name)
        elif hasattr(self.model, name):
            return getattr(self.model, name)
        elif hasattr(self.les, name):
            return getattr(self.les, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' has no atrribute '{name}'")
    @classmethod
    def from_json(cls, json_file):
        with open(json_file, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
