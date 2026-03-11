import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import vmap, jit, ops
from typing import Optional, Tuple, NamedTuple, List
import math
import time
class GraphData(NamedTuple):
    pos: jnp.ndarray
    batch: jnp.ndarray
    z: jnp.ndarray
    natoms: jnp.ndarray
    edge_index: jnp.ndarray
    edge_attr: jnp.ndarray
    edge_vec: jnp.ndarray
    cell: Optional[jnp.ndarray] = None
    cell_offsets: Optional[jnp.ndarray] = None
    displacement: Optional[jnp.ndarray] = None
    pbc: Optional[jnp.ndarray] = None
    atom_mask: Optional[jnp.ndarray] = None
    edge_mask: Optional[jnp.ndarray] = None

def segment_coo(src, index, dim_size):
    return ops.segment_sum(src, index, indices_are_sorted=True, num_segments=dim_size)

def segment_csr(src, indptr):
    indices = jnp.zeros(indptr[-1], dtype=int)
    for i in range(len(indptr)-1):
        indices = indices.at[indptr[i]:indptr[i+1]].set(i)
    return ops.segment_sum(src, indices, num_segments=len(indptr)-1)

def get_max_neighbors_mask(
    natoms: jnp.ndarray,
    index: jnp.ndarray,
    atom_distance: jnp.ndarray,
    max_num_neighbors_threshold: int,
    precision: jnp.dtype = jnp.float32
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    num_atoms = jnp.sum(natoms)
    ones = jnp.ones_like(index)
    num_neighbors = segment_coo(ones, index, dim_size=num_atoms)
    max_num_neighbors = jnp.max(num_neighbors).astype(jnp.int32)
    num_neighbors_thresholded = jnp.minimum(num_neighbors, max_num_neighbors_threshold)

    image_indptr = jnp.zeros(len(natoms) + 1, dtype=jnp.int32)
    image_indptr = image_indptr.at[1:].set(jnp.cumsum(natoms))
    num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    early_return = (max_num_neighbors <= max_num_neighbors_threshold) | (max_num_neighbors_threshold <= 0)
    if early_return:
        return jnp.ones_like(index, dtype=jnp.bool_), num_neighbors_image

    index_neighbor_offset = jnp.cumsum(num_neighbors) - num_neighbors
    index_neighbor_offset_expand = jnp.repeat(index_neighbor_offset, num_neighbors)

    index_sort_map = (
        index * max_num_neighbors_threshold +
        jnp.arange(len(index)) - 
        index_neighbor_offset_expand
    )
    

    max_possible_size = num_atoms * max_num_neighbors_threshold
    distance_sort = jnp.full(max_possible_size, jnp.inf, dtype=precision)

    valid_mask = index_sort_map < max_possible_size
    index_sort_map_safe = jnp.where(valid_mask, index_sort_map, 0)
    distance_sort = distance_sort.at[index_sort_map_safe].set(
        jnp.where(valid_mask, atom_distance, jnp.inf)
    )

    distance_sort = distance_sort.reshape(num_atoms, max_num_neighbors_threshold)

    sorted_indices = jnp.argsort(distance_sort, axis=1)
    sorted_distances = jnp.take_along_axis(distance_sort, sorted_indices, axis=1)

    sorted_indices = sorted_indices[:, :max_num_neighbors_threshold]

    sorted_indices = sorted_indices + jnp.expand_dims(index_neighbor_offset, axis=1)
 
    mask_finite = jnp.isfinite(sorted_distances)
    valid_indices = jnp.where(mask_finite, sorted_indices, -1)
    valid_indices = valid_indices.flatten()
    valid_indices = valid_indices[valid_indices != -1]

    mask_num_neighbors = jnp.zeros_like(index, dtype=jnp.bool_)
    mask_num_neighbors = mask_num_neighbors.at[valid_indices].set(True)
    
    return mask_num_neighbors, num_neighbors_image

def check_and_reshape_cell(cell: Optional[jnp.ndarray]) -> jnp.ndarray:
    if cell is None:
        return jnp.eye(3, dtype=jnp.float32)[jnp.newaxis, :, :]
    
    if cell.ndim == 2 and cell.shape[0] % 3 == 0 and cell.shape[1] == 3:
        batch_size = cell.shape[0] // 3
        return cell.reshape(batch_size, 3, 3)
    elif cell.ndim != 3 or cell.shape[1] != 3 or cell.shape[2] != 3:
        raise ValueError(f"Invalid cell shape. Expected (batch_size, 3, 3), but got {cell.shape}")
    return cell

def radius_graph_pbc(
    pos: jnp.ndarray,
    natoms: jnp.ndarray,
    cell: jnp.ndarray, 
    radius: float,
    max_num_neighbors_threshold: int,
    pbc: Optional[List[bool]] = None,
    precision: jnp.dtype = jnp.float32
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if pbc is None:
        pbc = [True, True, True]
    
    batch_size = len(natoms)
    num_atoms_per_image = natoms
    num_atoms_per_image_sqr = (num_atoms_per_image ** 2).astype(jnp.int32)

    index_offset = jnp.concatenate([jnp.array([0]), jnp.cumsum(num_atoms_per_image)[:-1]])
    index_offset_expand = jnp.repeat(index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = jnp.repeat(num_atoms_per_image, num_atoms_per_image_sqr)

    num_atom_pairs = jnp.sum(num_atoms_per_image_sqr)
    index_sqr_offset = jnp.concatenate([jnp.array([0]), jnp.cumsum(num_atoms_per_image_sqr)[:-1]])
    index_sqr_offset = jnp.repeat(index_sqr_offset, num_atoms_per_image_sqr)
    atom_count_sqr = jnp.arange(num_atom_pairs) - index_sqr_offset

    index1 = jnp.floor_divide(atom_count_sqr, num_atoms_per_image_expand) + index_offset_expand
    index2 = jnp.mod(atom_count_sqr, num_atoms_per_image_expand) + index_offset_expand

    pos1 = pos[index1]
    pos2 = pos[index2]

    cross_a2a3 = jnp.cross(cell[:, 1], cell[:, 2])
    cell_vol = jnp.sum(cell[:, 0] * cross_a2a3, axis=-1, keepdims=True)
    
    rep = []
    if pbc[0]:
        inv_min_dist_a1 = jnp.linalg.norm(cross_a2a3 / cell_vol, axis=-1)
        rep.append(jnp.ceil(radius * inv_min_dist_a1))
    else:
        rep.append(jnp.zeros(1, dtype=precision))
    
    if pbc[1]:
        cross_a3a1 = jnp.cross(cell[:, 2], cell[:, 0])
        inv_min_dist_a2 = jnp.linalg.norm(cross_a3a1 / cell_vol, axis=-1)
        rep.append(jnp.ceil(radius * inv_min_dist_a2))
    else:
        rep.append(jnp.zeros(1, dtype=precision))
    
    if pbc[2]:
        cross_a1a2 = jnp.cross(cell[:, 0], cell[:, 1])
        inv_min_dist_a3 = jnp.linalg.norm(cross_a1a2 / cell_vol, axis=-1)
        rep.append(jnp.ceil(radius * inv_min_dist_a3))
    else:
        rep.append(jnp.zeros(1, dtype=precision))
    
    max_rep = [int(jnp.max(r)) for r in rep]

    cells_per_dim = [
        jnp.arange(-r, r + 1, dtype=precision) for r in max_rep
    ]
    unit_cell = jnp.stack(jnp.meshgrid(*cells_per_dim, indexing='ij'), axis=-1).reshape(-1, 3)
    num_cells = len(unit_cell)

    unit_cell_per_atom = jnp.tile(unit_cell[jnp.newaxis, :, :], (len(index2), 1, 1))
    unit_cell_batch = jnp.tile(unit_cell.T[jnp.newaxis, :, :], (batch_size, 1, 1))

    data_cell = jnp.transpose(cell, (0, 2, 1))
    pbc_offsets = jnp.einsum('bij,bjk->bik', data_cell, unit_cell_batch)
    pbc_offsets_per_atom = jnp.repeat(pbc_offsets, num_atoms_per_image_sqr, axis=0)

    pos1 = jnp.tile(pos1[:, :, jnp.newaxis], (1, 1, num_cells))
    pos2 = jnp.tile(pos2[:, :, jnp.newaxis], (1, 1, num_cells)) + pbc_offsets_per_atom
    index1_exp = jnp.tile(index1[:, jnp.newaxis], (1, num_cells)).flatten()
    index2_exp = jnp.tile(index2[:, jnp.newaxis], (1, num_cells)).flatten()

    diff = pos1 - pos2
    atom_distance_sqr = jnp.sum(diff**2, axis=1).flatten()

    mask_within_radius = atom_distance_sqr <= radius**2
    mask_not_same = atom_distance_sqr > 0.0001
    mask = mask_within_radius & mask_not_same

    index1_masked = index1_exp[mask]
    index2_masked = index2_exp[mask]
    unit_cell_masked = unit_cell_per_atom.reshape(-1, 3)[mask]
    atom_distance_sqr_masked = atom_distance_sqr[mask]

    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms, index1_masked, atom_distance_sqr_masked, max_num_neighbors_threshold, precision
    )

    index1_final = index1_masked[mask_num_neighbors]
    index2_final = index2_masked[mask_num_neighbors]
    unit_cell_final = unit_cell_masked[mask_num_neighbors]
    
    edge_index = jnp.stack([index2_final, index1_final], axis=0)
    
    return edge_index, unit_cell_final, num_neighbors_image


def get_pbc_distances(
    pos: jnp.ndarray,
    edge_index: jnp.ndarray,
    cell: jnp.ndarray,
    cell_offsets: jnp.ndarray,
    edge_mask: Optional[jnp.ndarray],
    cutoff: float,
    precision: jnp.ndarray.dtype,
) -> dict:
    row = edge_index[0]
    col = edge_index[1]
    distance_vectors = pos[row] - pos[col]

    num_edges = row.shape[0]
    if cell.shape[0] != 1:
        raise ValueError("Padded JAX inference only supports a single graph per call.")
    cell_repeat = jnp.broadcast_to(cell[0], (num_edges, 3, 3))
    offsets = jnp.einsum('ij,ijk->ik', cell_offsets.astype(precision), cell_repeat.astype(precision))

    if edge_mask is None:
        edge_mask = jnp.ones((num_edges,), dtype=jnp.bool_)
    else:
        edge_mask = edge_mask.astype(jnp.bool_)

    distance_vectors += offsets
    safe_fill = jnp.broadcast_to(
        jnp.array([cutoff, 0.0, 0.0], dtype=precision),
        distance_vectors.shape,
    )
    distance_vectors_for_norm = jnp.where(edge_mask[:, None], distance_vectors, safe_fill)
    distances = jnp.linalg.norm(distance_vectors_for_norm, axis=-1)
    distance_vectors = jnp.where(edge_mask[:, None], distance_vectors, 0.0)

    result = {
        "edge_index": edge_index,
        "distances": distances,
    }

    result["distance_vec"] = distance_vectors
    return result

def get_symmetric_displacement(
    positions: jnp.ndarray,
    cell: Optional[jnp.ndarray],
    num_graphs: int,
    batch: jnp.ndarray,
    displacement: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if cell is None:
        cell = jnp.zeros((num_graphs * 3, 3), dtype=positions.dtype)
    
    #displacement = jnp.zeros((num_graphs, 3, 3), dtype=positions.dtype)
    symmetric_displacement = 0.5 * (displacement + jnp.transpose(displacement, (0, 2, 1)))

    positions = positions + jnp.einsum('be,bec->bc', positions, symmetric_displacement[batch])
    
    if cell is not None:
        cell_3d = cell.reshape(-1, 3, 3)
        cell_3d = cell_3d + jnp.einsum('bij,bjk->bik', cell_3d, symmetric_displacement)
        cell = cell_3d.reshape(-1, 3)
    
    return positions, cell#, displacement

def process_positions_and_edges(
    pos: jnp.ndarray,
    z: jnp.ndarray,
    natoms: jnp.ndarray,
    batch: jnp.ndarray,
    edge_index: jnp.ndarray,
    shift: jnp.ndarray,
    cell: Optional[jnp.ndarray] = None,
    displacement: Optional[jnp.ndarray] = None,
    atom_mask: Optional[jnp.ndarray] = None,
    edge_mask: Optional[jnp.ndarray] = None,
    compute_stress: bool = False,
    compute_forces: bool = False,
    use_pbc: bool = True,
    cutoff: float = 5.0,
    dtype: jnp.dtype = jnp.float32
) -> GraphData:

    
    precision = dtype
    pos = pos.astype(precision)
    z = z.astype(jnp.int32)
    batch = batch.astype(jnp.int32)
    edge_index = edge_index.astype(jnp.int32)
    shift = shift.astype(jnp.int32)
    if displacement is None:
        displacement = jnp.zeros((1, 3, 3), dtype=precision)
    if atom_mask is None:
        atom_mask = jnp.ones((pos.shape[0],), dtype=jnp.bool_)
    else:
        atom_mask = atom_mask.astype(jnp.bool_)
    if edge_mask is None:
        edge_mask = jnp.ones((edge_index.shape[1],), dtype=jnp.bool_)
    else:
        edge_mask = edge_mask.astype(jnp.bool_)

    pos, cell = get_symmetric_displacement(pos, cell, 1, batch, displacement)
    pos = jnp.where(atom_mask[:, None], pos, 0.0)
    z = jnp.where(atom_mask, z, 0)

    cell = check_and_reshape_cell(cell)
    if use_pbc and cell is not None:

        out = get_pbc_distances(
            pos,
            edge_index,
            cell,
            shift,
            edge_mask=edge_mask,
            cutoff=cutoff,
            precision=precision
        )
        edge_index = out["edge_index"]
        dist = out["distances"]
        vecs = out["distance_vec"]
    else:
        raise ValueError("None PBC is not supporting yet")
    

    return GraphData(
        pos=pos,
        z=z,
        natoms=natoms,
        batch=batch,
        edge_index=edge_index,
        edge_attr=dist,
        edge_vec=vecs,
        cell=cell,
        cell_offsets=shift,
        displacement=displacement,
        atom_mask=atom_mask,
        edge_mask=edge_mask,
    )
