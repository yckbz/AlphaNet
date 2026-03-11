#!/usr/bin/env python3
import os
import sys
import logging
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
# from flax import linen as nn  # No longer needed, as dummy checkpoint is removed
from typing import Optional, Dict, Any, Callable
# import orbax.checkpoint as orbax # <-- Changed to legacy serialization
from flax import serialization # <-- [IMPORTANT] Switched to legacy flax.serialization
import pickle
from dataclasses import dataclass
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from alphanet.models.alpha_flax import AlphaNet_flax as AlphaNet_flax
from alphanet.models.alpha_haiku import AlphaNet_hiku as AlphaNet_hiku

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Placeholder Import for All_Config (mimicking conv_pt2flax.py) ---
try:
    from alphanet.config import All_Config
    logger.info("Successfully imported 'alphanet.config.All_Config'")
except ImportError:
    logger.warning("Could not import 'All_Config' from 'alphanet.config'. Using a mock placeholder class.")
    
    class All_Config:
        """
        A mock placeholder for All_Config.
        Implements the .from_json() classmethod to mimic the behavior
        of the conv_pt2flax.py script, including loading, parsing the "model" key,
        and fallback logic.
        """
        def __init__(self, config_dict: Dict):
            if config_dict is None:
                raise ValueError("Config dictionary cannot be None.")
            for key, value in config_dict.items():
                setattr(self, key, value)
            
            # Ensure 'zbl' attribute exists (based on previous flax setup code)
            if 'zbl' not in config_dict:
                self.zbl = False
        
        @classmethod
        def from_json(cls, json_path: str):
            logger.info(f"(Placeholder) Loading configuration from {json_path}...")
            if not os.path.exists(json_path):
                logger.error(f"Configuration file not found: {json_path}")
                # (Fallback dictionary from conv_pt2flax.py)
                model_config_dict = {
                    "hidden_channels": 176, "num_radial": 8, "cutoff": 5.0, "num_layers": 4,
                    "head": 16, "chi1": 24, "chi2": 6, "mp_chi1": 24, "hidden_channels_chi": 96,
                    "dtype": "64", "eps": 1e-8, "a": 1, "b": 1, "readout": 'sum',
                    "use_sigmoid": False, "output_dim": 1, "compute_forces": True,
                    "compute_stress": True, "main_chi1": 24, "has_dropout_flag": False,
                    "has_norm_before_flag": True, "has_norm_after_flag": False, "reduce_mode": 'sum',
                    "zbl": False # Ensure zbl is in the fallback
                }
                logger.warning("Using default fallback configuration for demonstration.")
            else:
                with open(json_path, 'r') as f:
                    config_data = json.load(f)
                # (Logic from conv_pt2flax.py)
                model_config_dict = config_data.get("model")
                if model_config_dict is None:
                    raise ValueError(f"The file '{json_path}' must contain a 'model' key with the model config.")
            
            # Return an All_Config instance
            return cls(model_config_dict)

# --- Dummy Data (from flax2haiku.py) ---
# (Still needed to initialize the Haiku model structure)
@dataclass
class MoleculeData:
    pos: jnp.ndarray
    batch: jnp.ndarray
    z: jnp.ndarray
    edge_index: jnp.ndarray
    edge_attr: jnp.ndarray
    edge_vec: jnp.ndarray
    atom_mask: Optional[jnp.ndarray] = None
    edge_mask: Optional[jnp.ndarray] = None
    shift: Optional[jnp.ndarray] = None
    cell: Optional[jnp.ndarray] = None
    # ... (__post_init__ omitted for brevity)

def create_dummy_data(num_atoms=10, num_edges=20, dtype=jnp.float32, periodic=False):
    pos = jnp.array(np.random.randn(num_atoms, 3), dtype=dtype)
    batch = jnp.zeros(num_atoms, dtype=jnp.int32)
    z = jnp.array(np.random.randint(1, 95, num_atoms), dtype=jnp.int32)
    edge_index = np.zeros((2, num_edges), dtype=np.int32)
    for i in range(num_edges):
        edge_index[0, i] = np.random.randint(0, num_atoms)
        edge_index[1, i] = np.random.randint(0, num_atoms)
    edge_index = jnp.array(edge_index)
    edge_attr = jnp.array(np.random.rand(num_edges), dtype=dtype)
    edge_vec = jnp.array(np.random.randn(num_edges, 3), dtype=dtype)
    atom_mask = jnp.ones((num_atoms,), dtype=jnp.bool_)
    edge_mask = jnp.ones((num_edges,), dtype=jnp.bool_)
    shift = jnp.array(np.random.randn(num_edges, 3), dtype=dtype) if periodic else None
    cell = jnp.array(np.random.randn(3, 3), dtype=dtype) if periodic else None
    
    return MoleculeData(
        pos=pos, batch=batch, z=z, edge_index=edge_index,
        edge_attr=edge_attr, edge_vec=edge_vec,
        atom_mask=atom_mask, edge_mask=edge_mask,
        shift=shift, cell=cell
    )

class FlaxToHaikuConverter:
    """
    Converter for transforming a Flax checkpoint into Haiku weights.
    Mimics the structure of conv_pt2flax.py.
    """
    def __init__(self, haiku_model_fn: Callable, flax_path: str, haiku_output_dir: str, config: All_Config):
        """
        Args:
            haiku_model_fn: A function that, when called by hk.transform,
                            returns a Haiku model instance.
            flax_path: The Flax checkpoint *directory* containing 'flax_model.ckpt'.
            haiku_output_dir: Directory to save the converted Haiku weights (.pkl) and logs.
            config: A Config class instance containing model hyperparameters.
        """
        self.haiku_model_fn = haiku_model_fn
        self.flax_path = flax_path
        self.output_dir = haiku_output_dir
        self.config = config
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.mapping = self._create_mapping()

    def _load_flax_state_dict(self) -> Dict:
        """
        [Modified] Loads Flax weights from a legacy Flax serialization .ckpt file.
        Compatible with _save_flax_checkpoint from conv_pt2flax.py.
        """
        # self.flax_path is the directory (e.g., './flax_model_converted')
        # The target file is 'flax_model.ckpt'
        ckpt_file_path = os.path.join(self.flax_path, 'flax_model.ckpt')

        if not os.path.exists(ckpt_file_path):
            raise FileNotFoundError(f"Flax checkpoint file not found: {ckpt_file_path}\n"
                                  f"Please ensure you have run conv_pt2flax.py first and it successfully created the .ckpt file at this location.")
        
        logger.info(f"Loading (legacy) Flax weights from: {ckpt_file_path}")
        
        try:
            with open(ckpt_file_path, 'rb') as f:
                state_bytes = f.read()
            
            # Use None as target, as we just want to restore the PyTree from bytes
            restored_params = serialization.from_bytes(None, state_bytes)
            
            logger.info("Flax (legacy) checkpoint loaded successfully.")
            return restored_params
        except Exception as e:
            logger.error(f"Failed to load {ckpt_file_path}. It might be empty or corrupted. Error: {e}", exc_info=True)
            raise

    def _create_mapping(self):
        """Creates the static mapping from Flax keys to Haiku keys."""
        # (This function remains unchanged)
        mapping = {
            'params/z_emb/embedding': 'alpha_net_hiku/~/embed/embeddings',
            'params/radial_emb/bessel_weights': 'alpha_net_hiku/~/rbf_emb/bessel_weights',
            'params/radial_lin/layers_0/kernel': 'alpha_net_hiku/~/linear/w',
            'params/radial_lin/layers_0/bias':   'alpha_net_hiku/~/linear/b',
            'params/radial_lin/layers_2/kernel': 'alpha_net_hiku/~/linear_1/w',
            'params/radial_lin/layers_2/bias':   'alpha_net_hiku/~/linear_1/b',
            'params/neighbor_emb/Embed_0/embedding': 'alpha_net_hiku/~/neighbor_emb/embed/embeddings',
            'params/s_vector/Dense_0/kernel': 'alpha_net_hiku/~/s_vector/linear/w',
            'params/s_vector/Dense_0/bias':   'alpha_net_hiku/~/s_vector/linear/b',
            'params/lin/layers_0/kernel': 'alpha_net_hiku/~/linear_2/w',
            'params/lin/layers_0/bias':   'alpha_net_hiku/~/linear_2/b',
            'params/lin/layers_2/kernel': 'alpha_net_hiku/~/linear_3/w',
            'params/lin/layers_2/bias':   'alpha_net_hiku/~/linear_3/b',
            'params/kernel1':      'alpha_net_hiku/kernel1',
            'params/kernels_real': 'alpha_net_hiku/kernels_real',
            'params/kernels_imag': 'alpha_net_hiku/kernels_imag',
            'params/a': 'alpha_net_hiku/a',
            'params/b': 'alpha_net_hiku/b',
            'params/last_layer/kernel':          'alpha_net_hiku/~/linear_4/w',
            'params/last_layer/bias':            'alpha_net_hiku/~/linear_4/b',
            'params/last_layer_quantum/kernel':  'alpha_net_hiku/~/linear_5/w',
            'params/last_layer_quantum/bias':    'alpha_net_hiku/~/linear_5/b',
        }
        def suf(i): return "" if i == 0 else f"_{i}"
        for i in range(self.config.num_layers):
            emp = f'alpha_net_hiku/~/equi_message_passing{suf(i)}'
            fte = f'alpha_net_hiku/~/fte{suf(i)}'
            mapping.update({
                f'params/message_layers_{i}/x_layernorm/scale':         f'{emp}/layer_norm/scale',
                f'params/message_layers_{i}/x_layernorm/bias':          f'{emp}/layer_norm/offset',
                f'params/message_layers_{i}/x_proj/layers_0/kernel':    f'{emp}/~_build_x_proj/linear/w',
                f'params/message_layers_{i}/x_proj/layers_0/bias':      f'{emp}/~_build_x_proj/linear/b',
                f'params/message_layers_{i}/x_proj/layers_2/kernel':    f'{emp}/~_build_x_proj/linear_1/w',
                f'params/message_layers_{i}/x_proj/layers_2/bias':      f'{emp}/~_build_x_proj/linear_1/b',
                f'params/message_layers_{i}/rbf_proj/kernel':           f'{emp}/linear/w',
                f'params/message_layers_{i}/rbf_proj/bias':             f'{emp}/linear/b',
                f'params/message_layers_{i}/dir_proj/layers_0/kernel':  f'{emp}/~_build_dir_proj/linear/w',
                f'params/message_layers_{i}/dir_proj/layers_0/bias':    f'{emp}/~_build_dir_proj/linear/b',
                f'params/message_layers_{i}/dir_proj/layers_2/kernel':  f'{emp}/~_build_dir_proj/linear_1/w',
                f'params/message_layers_{i}/dir_proj/layers_2/bias':    f'{emp}/~_build_dir_proj/linear_1/b',
                f'params/message_layers_{i}/scale/kernel':              f'{emp}/~message/linear/w',
                f'params/message_layers_{i}/scale/bias':                f'{emp}/~message/linear/b',
                f'params/message_layers_{i}/kernel_real':               f'{emp}/kernel_real',
                f'params/message_layers_{i}/kernel_imag':               f'{emp}/kernel_imag',
                f'params/message_layers_{i}/diachi1':                   f'{emp}/diachi1',
                f'params/message_layers_{i}/diagonal/layers_0/kernel':  f'{emp}/~_build_diagonal/linear/w',
                f'params/message_layers_{i}/diagonal/layers_0/bias':    f'{emp}/~_build_diagonal/linear/b',
                f'params/message_layers_{i}/diagonal/layers_2/kernel':  f'{emp}/~_build_diagonal/linear_1/w',
                f'params/message_layers_{i}/diagonal/layers_2/bias':    f'{emp}/~_build_diagonal/linear_1/b',
                f'params/message_layers_{i}/dia/kernel':                f'{emp}/~message/linear_1/w',
                f'params/message_layers_{i}/dia/bias':                  f'{emp}/~message/linear_1/b',
                f'params/message_layers_{i}/fc_mps/kernel':             f'{emp}/~/linear/w',
                f'params/message_layers_{i}/fc_mps/bias':               f'{emp}/~/linear/b',
                f'params/message_layers_{i}/dx_layer_norm/scale':       f'{emp}/layer_norm_1/scale',
                f'params/message_layers_{i}/dx_layer_norm/bias':        f'{emp}/layer_norm_1/offset',
                f'params/message_layers_{i}/scale2/kernel':             f'{emp}/linear_1/w',
                f'params/message_layers_{i}/scale2/bias':               f'{emp}/linear_1/b',
            })
            mapping.update({
                f'params/ftes_{i}/vec_proj/kernel':                 f'{fte}/~/linear/w',
                f'params/ftes_{i}/xvec_proj/layers_0/kernel':       f'{fte}/~/linear_1/w',
                f'params/ftes_{i}/xvec_proj/layers_0/bias':         f'{fte}/~/linear_1/b',
                f'params/ftes_{i}/xvec_proj/layers_2/kernel':       f'{fte}/~/linear_2/w',
                f'params/ftes_{i}/xvec_proj/layers_2/bias':         f'{fte}/~/linear_2/b',
            })
        return mapping

    def _flatten_params(self, tree, prefix="") -> Dict[str, np.ndarray]:
        """Flattens a nested PyTree into a dict of {path: array}."""
        # (This function remains unchanged)
        leaves = {}
        if isinstance(tree, dict):
            for k, v in tree.items():
                new_prefix = f"{prefix}/{k}" if prefix else k
                leaves.update(self._flatten_params(v, new_prefix))
        elif hasattr(tree, "shape"):
            leaves[prefix] = np.asarray(tree) # Convert to NumPy
        return leaves

    def _unflatten_params(self, flat_params: Dict[str, np.ndarray]) -> Dict:
        """Converts a flat dict back into a Haiku-style nested dict."""
        # (This function remains unchanged)
        new_dict = {}
        for key, value in flat_params.items():
            parts = key.split('/')
            module_path = '/'.join(parts[:-1])
            if module_path not in new_dict:
                new_dict[module_path] = {}
            param_name = parts[-1]
            new_dict[module_path][param_name] = value
        return new_dict

    def _reshape_parameter(self, flax_key: str, array: np.ndarray) -> np.ndarray:
        """Reshapes parameters as needed."""
        # (This function remains unchanged)
        return np.asarray(array)

    def _log_haiku_structure(self, params, depth=0, prefix=""):
        """Recursively logs the structure of the Haiku model parameters."""
        # (This function remains unchanged)
        indent = "  " * depth
        if depth == 0:
            logger.info("--- Haiku Model Parameter Structure ---")
        
        if depth == 0:
            for module_path, param_dict in sorted(params.items()):
                logger.info(f"{indent}{module_path}/")
                self._log_haiku_structure(param_dict, depth + 1, f"{module_path}/")
        elif isinstance(params, dict):
             for key, value in sorted(params.items()):
                current_prefix = f"{prefix}{key}"
                if isinstance(value, dict):
                    logger.info(f"{indent}{current_prefix}/")
                    self._log_haiku_structure(value, depth + 1, f"{current_prefix}/")
                elif hasattr(value, 'shape'):
                    logger.info(f"{indent}{current_prefix}: {value.shape}")
        
        if depth == 0:
            logger.info("------------------------------------")

    def _log_conversion_results(self, conversion_map, missing_flax_keys, shape_mismatches, unfilled_haiku_keys):
        """Logs a summary of the conversion process to files."""
        # (This function remains unchanged, but logs are translated)
        with open(os.path.join(self.output_dir, "conversion_map.txt"), 'w') as f:
            f.write("Flax Key -> Haiku Key Mapping\n" + "-"*80 + "\n")
            for flax_key, haiku_key in sorted(conversion_map.items()):
                f.write(f"{flax_key:<60} -> {haiku_key}\n")

        if missing_flax_keys:
            logger.warning(f"{len(missing_flax_keys)} Flax keys could not be mapped or found in the target Haiku model.")
            with open(os.path.join(self.output_dir, "missing_flax_keys.txt"), 'w') as f:
                f.write(f"Unmapped or Superfluous Flax Keys ({len(missing_flax_keys)})\n" + "-"*80 + "\n")
                for flax_key, reason in sorted(missing_flax_keys):
                    f.write(f"Flax: {flax_key} -> Reason: {reason}\n")
        
        if shape_mismatches:
            logger.warning(f"{len(shape_mismatches)} parameters had shape mismatches.")
            with open(os.path.join(self.output_dir, "shape_mismatches.txt"), 'w') as f:
                f.write(f"Shape Mismatches ({len(shape_mismatches)})\n" + "-"*100 + "\n")
                header = f"{'Flax Key':<50} | {'Haiku Key':<50} | {'Flax Shape':<20} | {'Haiku Shape':<20}\n"
                f.write(header)
                f.write("-" * (len(header) + 5) + "\n")
                for flax_key, haiku_key, flax_shape, haiku_shape in sorted(shape_mismatches):
                    f.write(f"{flax_key:<50} | {haiku_key:<50} | {str(flax_shape):<20} | {str(haiku_shape):<20}\n")
        
        if unfilled_haiku_keys:
            logger.warning(f"{len(unfilled_haiku_keys)} Haiku keys were not filled by Flax weights (will use init values).")
            with open(os.path.join(self.output_dir, "unfilled_haiku_keys.txt"), 'w') as f:
                f.write(f"Unfilled Haiku Keys ({len(unfilled_haiku_keys)})\n" + "-"*80 + "\n")
                for haiku_key in sorted(unfilled_haiku_keys):
                    f.write(f"{haiku_key}\n")

        if not missing_flax_keys and not shape_mismatches and not unfilled_haiku_keys:
            logger.info("Conversion successful! All parameters mapped with no errors.")
        else:
            logger.warning("Conversion complete, but issues were found. Please check the log files in the output directory.")

    def _save_haiku_checkpoint(self, state_dict: Dict):
        """Saves the converted Haiku state dict to a pickle file."""
        # (This function remains unchanged)
        output_path = os.path.join(self.output_dir, 'haiku_model.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(state_dict, f)
        logger.info(f"Haiku model checkpoint saved successfully to: {output_path}")

    def convert(self) -> Dict:
        """
        Executes the full conversion process from Flax to Haiku.
        """
        # (This function remains unchanged)
        # 1. Load source (Flax) weights and flatten
        flax_state_dict_nested = self._load_flax_state_dict()
        flax_state_dict_flat = self._flatten_params(flax_state_dict_nested)
        
        # 2. Initialize target (Haiku) model to get structure
        logger.info("Initializing Haiku model to get parameter structure...")
        rng = jax.random.PRNGKey(42)
        dtype = jnp.float64 if self.config.dtype == "64" else jnp.float32
        dummy_data = create_dummy_data(
            num_atoms=5, num_edges=8, dtype=dtype, periodic=False
        )
        
        transformed = hk.transform(self.haiku_model_fn)
        haiku_params_init_nested = transformed.init(rng, dummy_data)
        haiku_params_init_flat = self._flatten_params(haiku_params_init_nested)
        
        self._log_haiku_structure(haiku_params_init_nested)

        # 3. Prepare for conversion
        haiku_params_converted_flat = {}
        conversion_map = {}
        missing_flax_keys = []
        shape_mismatches = []
        unfilled_haiku_keys = set(haiku_params_init_flat.keys()) # Track Haiku keys that are not covered

        # 4. Iterate over *source (Flax)* parameters and convert
        for flax_key, flax_value in flax_state_dict_flat.items():
            haiku_key = self.mapping.get(flax_key)
            
            if haiku_key is None:
                missing_flax_keys.append((flax_key, "Not found in static mapping"))
                continue

            conversion_map[flax_key] = haiku_key
            haiku_value = self._reshape_parameter(flax_key, flax_value)

            if haiku_key not in haiku_params_init_flat:
                missing_flax_keys.append((flax_key, f"Mapped Haiku key '{haiku_key}' not in target model"))
                continue
            
            target_shape = haiku_params_init_flat[haiku_key].shape
            if target_shape != haiku_value.shape:
                shape_mismatches.append((flax_key, haiku_key, haiku_value.shape, target_shape))
                continue
            
            haiku_params_converted_flat[haiku_key] = haiku_value
            unfilled_haiku_keys.discard(haiku_key) # Mark this key as filled

        # 5. Merge parameters
        final_haiku_params_flat = haiku_params_init_flat.copy()
        final_haiku_params_flat.update(haiku_params_converted_flat)
        
        # 6. Log results
        self._log_conversion_results(
            conversion_map, 
            missing_flax_keys, 
            shape_mismatches, 
            unfilled_haiku_keys
        )

        # 7. Unflatten and save
        final_haiku_state_dict_nested = self._unflatten_params(final_haiku_params_flat)
        self._save_haiku_checkpoint(final_haiku_state_dict_nested)
        
        return final_haiku_state_dict_nested

#
# [Removed] The generate_dummy_flax_checkpoint function is no longer needed
#

if __name__ == "__main__":
    
    # 1. Define configuration and paths
    config_path = "./pretrained/MATPES/matpes.json"
    
    # [IMPORTANT] FLAX_CHECKPOINT_DIR is now an *input* path
    # It should match the flax_output_dir from conv_pt2flax.py
    FLAX_CHECKPOINT_DIR = "./flax_model_converted"
    HAIKU_OUTPUT_DIR = "./haiku_model_converted"

    # 2. Initialize model configuration
    try:
        config = All_Config.from_json(config_path)
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}", exc_info=True)
        sys.exit(1)

    try:
        #
        # [Removed] No longer calling generate_dummy_flax_checkpoint
        #
        logger.info(f"Preparing to load Flax checkpoint from {FLAX_CHECKPOINT_DIR}...")

        # 3. Define the Haiku model constructor function
        def haiku_fn(data):
            model = AlphaNet_hiku(config) # Pass the config instance
            return model(data)
        
        # 4. Instantiate the converter
        converter = FlaxToHaikuConverter(
            haiku_model_fn=haiku_fn,
            flax_path=FLAX_CHECKPOINT_DIR, # <-- Pass the *existing* checkpoint directory
            haiku_output_dir=HAIKU_OUTPUT_DIR,
            config=config # Pass the config instance
        )
        
        # 5. Run the conversion process
        logger.info("--- Starting Flax-to-Haiku Conversion ---")
        haiku_state_dict = converter.convert()
        logger.info("Conversion process finished.")

    except FileNotFoundError as e:
        logger.error(f"File Not Found Error: {e}")
        logger.error("Please ensure you have already run the PyTorch-to-Flax conversion script (conv_pt2flax.py) "
                     f"and that the 'flax_model.ckpt' file exists in the '{FLAX_CHECKPOINT_DIR}' directory.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during conversion: {e}", exc_info=True)
        sys.exit(1)
