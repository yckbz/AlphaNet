import pickle

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers
from jax import value_and_grad
from matscipy.neighbours import neighbour_list

from alphanet.models.alpha_haiku import AlphaNet_hiku
from alphanet.models.graph_jax import process_positions_and_edges


DEFAULT_ATOM_BUCKETS = (
    64,
    96,
    128,
    160,
    192,
    256,
    320,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    10240,
)


class AlphaNetCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        ckpt_path,
        config,
        device="cpu",
        precision="32",
        max_atoms=None,
        max_num_neighbors=50,
        atom_buckets=None,
        neighbor_slack=4,
        neighbor_bucket_multiple=4,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.config = config
        self.precision = jnp.float32 if precision == "32" else jnp.float64
        self.dtype = np.float32 if precision == "32" else np.float64
        self.max_atoms = None if max_atoms is None else int(max_atoms)
        self.max_num_neighbors = int(max_num_neighbors)
        if self.max_num_neighbors <= 0:
            raise ValueError("max_num_neighbors must be a positive integer.")
        self.atom_buckets = tuple(int(x) for x in (atom_buckets or DEFAULT_ATOM_BUCKETS))
        self.neighbor_slack = int(neighbor_slack)
        self.neighbor_bucket_multiple = int(neighbor_bucket_multiple)
        self._last_runtime_plan = None

        self.device = jax.devices("cpu")[0] if device == "cpu" else jax.devices("gpu")[0]
        self.rng = jax.random.PRNGKey(0)

        def forward_fn(graph_data):
            model = AlphaNet_hiku(config)
            return model(graph_data)

        self.transform = hk.transform(forward_fn)
        graph_data = create_dummy_data()
        init_params = self.transform.init(self.rng, graph_data)

        if ckpt_path.endswith("ckpt") or ckpt_path.endswith("pkl"):
            with open(ckpt_path, "rb") as f:
                raw_params = pickle.load(f)
        else:
            raise ValueError("Unknown checkpoint format")

        base_leaves = flatten_leaf_arrays(init_params)
        override_leaves = flatten_leaf_arrays(raw_params)
        check_params(override_leaves, base_leaves)
        merged_leaves = base_leaves.copy()
        merged_leaves.update(override_leaves)
        self.params = jax.device_put(convert_dict_keys(merged_leaves), self.device)

        self._compiled_functions = {}

    def _get_capacity(self, num_atoms):
        if self.max_atoms is not None:
            if num_atoms > self.max_atoms:
                raise ValueError(
                    f"Current structure has {num_atoms} atoms, exceeding max_atoms={self.max_atoms}."
                )
            return self.max_atoms

        for capacity in self.atom_buckets:
            if capacity >= num_atoms:
                return capacity

        capacity = self.atom_buckets[-1]
        while capacity < num_atoms:
            capacity *= 2
        return capacity

    def _round_up_neighbors(self, value):
        if self.neighbor_bucket_multiple <= 1:
            return int(value)
        multiple = self.neighbor_bucket_multiple
        return ((int(value) + multiple - 1) // multiple) * multiple

    def _compile_energy_and_grad_fn(self, max_atoms, max_edges):
        batch = jnp.zeros((max_atoms,), dtype=jnp.int32)

        def energy_fn(pos, z, atom_mask, natoms, cell, edge_index, shift, edge_mask, displacement):
            graph_data = process_positions_and_edges(
                pos=pos,
                z=z,
                natoms=natoms,
                batch=batch,
                cell=cell,
                edge_index=edge_index,
                shift=shift,
                displacement=displacement,
                atom_mask=atom_mask,
                edge_mask=edge_mask,
                cutoff=self.config.cutoff,
                dtype=self.precision,
            )
            return self.transform.apply(self.params, self.rng, graph_data)

        if self.config.compute_forces and self.config.compute_stress:
            grad_fn = jax.jit(
                value_and_grad(energy_fn, (0, 8)),
                device=self.device,
            )
        elif self.config.compute_forces:
            grad_fn = jax.jit(
                value_and_grad(energy_fn, 0),
                device=self.device,
            )
        else:
            grad_fn = jax.jit(energy_fn, device=self.device)

        return grad_fn

    def _prepare_atoms(self, atoms):
        calc_atoms = atoms.copy()
        if not calc_atoms.pbc.any():
            print("Non-periodic system detected. Automatically adding a large vacuum box for calculation.")
            padding = 20.0
            new_cell_dims = np.ptp(calc_atoms.get_positions(), axis=0) + padding
            calc_atoms.set_cell(np.diag(new_cell_dims))
            calc_atoms.center()
            calc_atoms.pbc = True
        return calc_atoms

    def _build_runtime_plan(self, atoms):
        index_i, index_j, distances, shifts = neighbour_list(
            quantities="ijdS",
            atoms=atoms,
            cutoff=self.config.cutoff,
        )
        counts = np.bincount(index_i, minlength=len(atoms)) if index_i.size else np.zeros(len(atoms), dtype=np.int32)
        observed_max_neighbors = int(counts.max()) if counts.size else 0
        if observed_max_neighbors > self.max_num_neighbors:
            raise ValueError(
                f"Observed max neighbors {observed_max_neighbors} exceed hard cap max_num_neighbors={self.max_num_neighbors}."
            )

        effective_neighbors = max(1, self._round_up_neighbors(observed_max_neighbors + self.neighbor_slack))
        effective_neighbors = min(effective_neighbors, self.max_num_neighbors)
        return {
            "index_i": index_i,
            "index_j": index_j,
            "distances": distances,
            "shifts": shifts,
            "observed_max_neighbors": observed_max_neighbors,
            "effective_max_num_neighbors": effective_neighbors,
        }

    def _trim_neighbor_list_with_limit(self, index_i, index_j, distances, shifts, max_num_neighbors):
        if index_i.size == 0 or max_num_neighbors <= 0:
            return index_i, index_j, shifts

        order = np.lexsort((distances, index_i))
        index_i = index_i[order]
        index_j = index_j[order]
        shifts = shifts[order]

        keep = np.zeros(index_i.shape[0], dtype=bool)
        last_center = -1
        center_count = 0
        for idx, center in enumerate(index_i):
            if center != last_center:
                last_center = center
                center_count = 0
            if center_count < max_num_neighbors:
                keep[idx] = True
                center_count += 1

        return index_i[keep], index_j[keep], shifts[keep]

    def _prepare_padded_inputs_from_plan(self, atoms, capacity_atoms, runtime_plan):
        effective_max_num_neighbors = runtime_plan["effective_max_num_neighbors"]
        max_edges = capacity_atoms * effective_max_num_neighbors
        num_atoms = len(atoms)

        z = np.zeros((capacity_atoms,), dtype=np.int32)
        z[:num_atoms] = [atomic_numbers[atom.symbol] for atom in atoms]

        pos = np.zeros((capacity_atoms, 3), dtype=self.dtype)
        pos[:num_atoms] = atoms.get_positions(wrap=True).astype(self.dtype, copy=False)

        atom_mask = np.zeros((capacity_atoms,), dtype=bool)
        atom_mask[:num_atoms] = True

        natoms = np.array([num_atoms], dtype=np.int32)
        cell = np.array(atoms.get_cell(complete=True), dtype=self.dtype, copy=True)[None, :, :]

        index_i, index_j, shifts = self._trim_neighbor_list_with_limit(
            runtime_plan["index_i"],
            runtime_plan["index_j"],
            runtime_plan["distances"],
            runtime_plan["shifts"],
            effective_max_num_neighbors,
        )

        if index_i.shape[0] > max_edges:
            raise ValueError(
                f"Neighbor list overflow: {index_i.shape[0]} edges exceed padded capacity {max_edges}."
            )

        edge_index = np.zeros((2, max_edges), dtype=np.int32)
        edge_shift = np.zeros((max_edges, 3), dtype=np.int32)
        edge_mask = np.zeros((max_edges,), dtype=bool)
        num_edges = index_i.shape[0]
        if num_edges:
            edge_index[0, :num_edges] = index_j.astype(np.int32, copy=False)
            edge_index[1, :num_edges] = index_i.astype(np.int32, copy=False)
            edge_shift[:num_edges] = shifts.astype(np.int32, copy=False)
            edge_mask[:num_edges] = True

        self._last_runtime_plan = {
            "num_atoms": num_atoms,
            "capacity_atoms": capacity_atoms,
            "observed_max_neighbors": runtime_plan["observed_max_neighbors"],
            "effective_max_num_neighbors": effective_max_num_neighbors,
            "num_edges": int(num_edges),
            "max_edges": int(max_edges),
        }

        return (
            jax.device_put(jnp.asarray(pos, dtype=self.precision), self.device),
            jax.device_put(jnp.asarray(z, dtype=jnp.int32), self.device),
            jax.device_put(jnp.asarray(atom_mask), self.device),
            jax.device_put(jnp.asarray(natoms, dtype=jnp.int32), self.device),
            jax.device_put(jnp.asarray(cell, dtype=self.precision), self.device),
            jax.device_put(jnp.asarray(edge_index, dtype=jnp.int32), self.device),
            jax.device_put(jnp.asarray(edge_shift, dtype=jnp.int32), self.device),
            jax.device_put(jnp.asarray(edge_mask), self.device),
        )

    def calculate(self, atoms=None, properties=None, system_changes=None):
        Calculator.calculate(self, atoms, properties, system_changes)
        properties = properties or ["energy"]
        calc_atoms = self._prepare_atoms(self.atoms)
        capacity_atoms = self._get_capacity(len(calc_atoms))
        runtime_plan = self._build_runtime_plan(calc_atoms)
        effective_max_num_neighbors = runtime_plan["effective_max_num_neighbors"]
        max_edges = capacity_atoms * effective_max_num_neighbors
        compile_key = (
            capacity_atoms,
            max_edges,
            self.precision,
            bool(self.config.compute_forces),
            bool(self.config.compute_stress),
        )
        if compile_key not in self._compiled_functions:
            self._compiled_functions[compile_key] = self._compile_energy_and_grad_fn(
                capacity_atoms,
                max_edges,
            )
        grad_fn = self._compiled_functions[compile_key]

        pos, z, atom_mask, natoms, cell, edge_index, shift, edge_mask = self._prepare_padded_inputs_from_plan(
            calc_atoms,
            capacity_atoms,
            runtime_plan,
        )
        displacement = jax.device_put(
            jnp.zeros((1, 3, 3), dtype=self.precision),
            self.device,
        )

        if self.config.compute_forces:
            if self.config.compute_stress:
                (energy, (minus_forces, pseudo_stress)) = grad_fn(
                    pos, z, atom_mask, natoms, cell, edge_index, shift, edge_mask, displacement
                )
            else:
                (energy, minus_forces) = grad_fn(
                    pos, z, atom_mask, natoms, cell, edge_index, shift, edge_mask, displacement
                )
                pseudo_stress = None
        else:
            energy = grad_fn(pos, z, atom_mask, natoms, cell, edge_index, shift, edge_mask, displacement)
            minus_forces = None
            pseudo_stress = None

        energy_np = np.asarray(energy)
        self.results["energy"] = energy_np.item() if energy_np.shape == () else energy_np
        self.results["free_energy"] = self.results["energy"]

        if minus_forces is not None:
            forces_np = -np.asarray(minus_forces)[:len(calc_atoms)]
            self.results["forces"] = forces_np

        if pseudo_stress is not None:
            stress = np.asarray(pseudo_stress) / calc_atoms.get_volume()
            stress_matrix = stress[0]
            self.results["stress"] = np.array(
                [
                    stress_matrix[0, 0],
                    stress_matrix[1, 1],
                    stress_matrix[2, 2],
                    0.5 * (stress_matrix[1, 2] + stress_matrix[2, 1]),
                    0.5 * (stress_matrix[0, 2] + stress_matrix[2, 0]),
                    0.5 * (stress_matrix[0, 1] + stress_matrix[1, 0]),
                ]
            )


def create_dummy_data():
    pos = jnp.array([[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]], dtype=jnp.float32)
    z = jnp.array([1, 1], dtype=jnp.int32)
    natoms = jnp.array([2], dtype=jnp.int32)
    batch = jnp.array([0, 0], dtype=jnp.int32)
    displacement = jnp.zeros((1, 3, 3), dtype=pos.dtype)
    atom_mask = jnp.array([True, True], dtype=jnp.bool_)
    cell = jnp.array(
        [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
        dtype=jnp.float32,
    )
    index_i, index_j, shift = neighbour_list(
        quantities="ijS",
        positions=np.asarray(pos),
        cell=np.asarray(cell[0]),
        pbc=np.array([True, True, True]),
        cutoff=1.0,
    )
    edge_index = jnp.zeros((2, 2), dtype=jnp.int32)
    edge_index = edge_index.at[0, : len(index_i)].set(jnp.asarray(index_j, dtype=jnp.int32))
    edge_index = edge_index.at[1, : len(index_i)].set(jnp.asarray(index_i, dtype=jnp.int32))
    edge_shift = jnp.zeros((2, 3), dtype=jnp.int32)
    edge_shift = edge_shift.at[: len(index_i)].set(jnp.asarray(shift, dtype=jnp.int32))
    edge_mask = jnp.zeros((2,), dtype=jnp.bool_)
    edge_mask = edge_mask.at[: len(index_i)].set(True)
    graph_data = process_positions_and_edges(
        pos=pos,
        z=z,
        natoms=natoms,
        batch=batch,
        cell=cell,
        edge_index=edge_index,
        shift=edge_shift,
        displacement=displacement,
        atom_mask=atom_mask,
        edge_mask=edge_mask,
    )
    return graph_data


def flatten_leaf_arrays(tree, prefix=""):
    leaves = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            new_prefix = f"{prefix}/{k}" if prefix else k
            leaves.update(flatten_leaf_arrays(v, new_prefix))
    else:
        if hasattr(tree, "shape"):
            leaves[prefix] = tree
    return leaves


def convert_dict_keys(old_dict):
    new_dict = {}
    for key, value in old_dict.items():
        parts = key.split("/")
        module_path = "/".join(parts[:-1])

        if module_path not in new_dict:
            new_dict[module_path] = {}

        param_name = parts[-1]
        new_dict[module_path][param_name] = value

    return new_dict


def check_params(override_leaves, base_leaves):
    shape_mismatch = []
    missing = sorted(set(base_leaves) - set(override_leaves))
    extra = sorted(set(override_leaves) - set(base_leaves))
    for k in override_leaves:
        if k in base_leaves and base_leaves[k].shape != override_leaves[k].shape:
            shape_mismatch.append((k, base_leaves[k].shape, override_leaves[k].shape))

    if missing:
        print("Missing leaf parameters")
        for p in missing[:50]:
            print("  ", p)
        raise ValueError(f"Total of {len(missing)} missing leaves.")

    if extra:
        print("Extra leaf parameters")
        for p in extra[:50]:
            print("  ", p)

    if shape_mismatch:
        for p, s0, s1 in shape_mismatch[:50]:
            print(f"{p} shape mismatch: {tuple(s0)} vs {tuple(s1)}")
        raise ValueError(f"Total of {len(shape_mismatch)} shape mismatches found.")
