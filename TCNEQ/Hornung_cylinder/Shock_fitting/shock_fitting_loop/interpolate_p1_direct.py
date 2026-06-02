#!/usr/bin/env python3
"""
DIRECT P1 INTERPOLATION: .msh + old P1 CFmesh -> new P1 CFmesh + R-H file
==========================================================================

Replaces the 3-step chain: interpolate_p0.py + MeshUpgrade + sf_adjust.py

Takes a split Gmsh .msh (from split_at_b23.py) and a converged P1 CFmesh,
writes a P1 CFmesh directly with nearest-state interpolated solution.

Algorithm:
  1. Parse split .msh -> nodes, quads, boundary faces
  2. Parse old P1 CFmesh -> nodes, elements, states, GL coordinates
  3. Create P1 layout: 4 GL states per quad (SOL_POLYORDER=1)
  4. Identify upstream domain (elements connected to ShockUp TRS)
  5. Downstream states: nearest-state KD-tree from old P1
  6. Upstream states: freestream
  7. Write P1 CFmesh
  8. Compute R-H post-shock state (PLATO) and write .dat file

Usage:
  LD_LIBRARY_PATH=.../plato.../lib:$LD_LIBRARY_PATH \\
  python3 interpolate_p1_direct.py \\
    --msh /path/to/shockfitting_v2.msh \\
    --old-solution /path/to/P1_CNEQ_SF-iter_XXXX.CFmesh \\
    --output-dir shock_fitting_loop/iter1
"""

import argparse
import os
import sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _sf_common import (
    parse_cfmesh_with_validation, identify_upstream_domain,
    compute_state_coordinates,
    compute_rh_all_faces, write_rh_file,
    plato_init, plato_finalize,
    FREESTREAM,
)

# ============================================================
# Gmsh .msh parser (split mesh with physical groups)
# ============================================================

def parse_split_msh(path):
    """Parse Gmsh v2.2 .msh with physical names. Returns 0-indexed data."""
    with open(path, 'r') as f:
        lines = f.readlines()

    phys_names = {}   # pid -> (dim, name)
    gmsh_nodes = {}   # gmsh_id -> (x, y)
    quads = []        # [(n0,n1,n2,n3), ...] in gmsh IDs
    boundary_lines = {} # pid -> [(nA, nB), ...] in gmsh IDs

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == '$PhysicalNames':
            i += 1
            n = int(lines[i].strip()); i += 1
            for _ in range(n):
                parts = lines[i].strip().split()
                dim, pid = int(parts[0]), int(parts[1])
                name = parts[2].strip('"')
                phys_names[pid] = (dim, name)
                i += 1
        elif line == '$Nodes':
            i += 1
            n = int(lines[i].strip()); i += 1
            for _ in range(n):
                parts = lines[i].strip().split()
                nid = int(parts[0])
                gmsh_nodes[nid] = (float(parts[1]), float(parts[2]))
                i += 1
        elif line == '$Elements':
            i += 1
            n = int(lines[i].strip()); i += 1
            for _ in range(n):
                parts = lines[i].strip().split()
                etype = int(parts[1])
                ntags = int(parts[2])
                pid = int(parts[3]) if ntags > 0 else 0
                if etype == 3:  # 4-node quad
                    nids = [int(parts[3 + ntags + j]) for j in range(4)]
                    quads.append(nids)
                elif etype == 1:  # 2-node line
                    nids = [int(parts[3 + ntags + j]) for j in range(2)]
                    if pid not in boundary_lines:
                        boundary_lines[pid] = []
                    boundary_lines[pid].append(tuple(nids))
                i += 1
        else:
            i += 1

    # Build 0-indexed mapping
    sorted_nids = sorted(gmsh_nodes.keys())
    nid_map = {nid: idx for idx, nid in enumerate(sorted_nids)}
    n_nodes = len(sorted_nids)

    # 0-indexed node coordinates
    nodes = np.zeros((n_nodes, 2))
    for nid in sorted_nids:
        nodes[nid_map[nid]] = gmsh_nodes[nid]

    # 0-indexed quads
    quads_0 = [[nid_map[n] for n in q] for q in quads]

    # 0-indexed boundary lines with TRS names
    trs_faces = {}  # name -> [(n0, n1), ...]
    for pid, (dim, name) in phys_names.items():
        if dim == 1 and pid in boundary_lines:
            trs_faces[name] = [(nid_map[a], nid_map[b])
                               for a, b in boundary_lines[pid]]

    return nodes, quads_0, trs_faces


# ============================================================
# P1 GL solution point coordinates
# ============================================================

_g = 1.0 / np.sqrt(3.0)
_GL_REF = [(-_g, -_g), (_g, -_g), (-_g, _g), (_g, _g)]


def compute_gl_coordinates(nodes, quads):
    """Compute physical (x,y) of 4 GL solution points per quad element."""
    n_cells = len(quads)
    n_states = 4 * n_cells
    coords = np.zeros((n_states, 2))

    for ci, q in enumerate(quads):
        x = np.array([nodes[n][0] for n in q])
        y = np.array([nodes[n][1] for n in q])
        for iSol in range(4):
            xi, eta = _GL_REF[iSol]
            N0 = 0.25 * (1 - xi) * (1 - eta)
            N1 = 0.25 * (1 + xi) * (1 - eta)
            N2 = 0.25 * (1 + xi) * (1 + eta)
            N3 = 0.25 * (1 - xi) * (1 + eta)
            sid = 4 * ci + iSol
            coords[sid, 0] = N0*x[0] + N1*x[1] + N2*x[2] + N3*x[3]
            coords[sid, 1] = N0*y[0] + N1*y[1] + N2*y[2] + N3*y[3]

    return coords


# ============================================================
# Upstream domain identification (from .msh topology)
# ============================================================

def identify_upstream_elements(quads, trs_faces):
    """
    Find upstream elements via flood fill from ShockUp boundary.
    Returns set of element indices.
    """
    if 'ShockUp' not in trs_faces:
        print("ERROR: ShockUp TRS not found in mesh")
        sys.exit(1)

    # Build node -> element adjacency
    node_to_elems = {}
    for ci, q in enumerate(quads):
        for n in q:
            if n not in node_to_elems:
                node_to_elems[n] = []
            node_to_elems[n].append(ci)

    # Seed: elements touching ShockUp boundary nodes
    seed_nodes = set()
    for nA, nB in trs_faces['ShockUp']:
        seed_nodes.add(nA)
        seed_nodes.add(nB)

    upstream = set()
    queue = set()
    for n in seed_nodes:
        for ci in node_to_elems.get(n, []):
            queue.add(ci)

    # Flood fill via shared nodes (within connected component)
    while queue:
        ci = queue.pop()
        if ci in upstream:
            continue
        upstream.add(ci)
        for n in quads[ci]:
            for neighbor in node_to_elems.get(n, []):
                if neighbor not in upstream:
                    queue.add(neighbor)

    return upstream


# ============================================================
# Write P1 CFmesh
# ============================================================

def write_p1_cfmesh(path, nodes, quads, trs_faces, states, nb_eq):
    """Write a P1 CFmesh file (SOL_POLYORDER=1, 4 states per quad)."""
    n_nodes = len(nodes)
    n_cells = len(quads)
    n_states = 4 * n_cells

    # TRS ordering: match what COOLFluiD expects
    trs_order = ['Inlet', 'Outlet', 'ShockDown', 'ShockUp', 'Wall']
    trs_list = []
    for name in trs_order:
        if name in trs_faces:
            trs_list.append((name, trs_faces[name]))
    # Add any remaining TRS not in the expected order
    for name in sorted(trs_faces.keys()):
        if name not in trs_order:
            trs_list.append((name, trs_faces[name]))

    with open(path, 'w') as f:
        f.write("!COOLFLUID_VERSION 2013.9\n")
        f.write("!CFMESH_FORMAT_VERSION 1.3\n")
        f.write("!NB_DIM 2\n")
        f.write(f"!NB_EQ {nb_eq}\n")
        f.write(f"!NB_NODES {n_nodes} 0\n")
        f.write(f"!NB_STATES {n_states} 0\n")
        f.write(f"!NB_ELEM {n_cells}\n")
        f.write("!NB_ELEM_TYPES 1\n")
        f.write("!GEOM_POLYORDER 1\n")
        f.write("!SOL_POLYORDER 1\n")
        f.write("!ELEM_TYPES Quad \n")
        f.write(f"!NB_ELEM_PER_TYPE {n_cells}\n")
        f.write("!NB_NODES_PER_TYPE 4\n")
        f.write("!NB_STATES_PER_TYPE 4\n")
        f.write("!LIST_ELEM \n")
        for ci, q in enumerate(quads):
            s0 = 4 * ci
            f.write(f"{q[0]} {q[1]} {q[2]} {q[3]} {s0} {s0+1} {s0+2} {s0+3} \n")

        f.write(f"!NB_TRSs {len(trs_list)}\n")
        for trs_name, faces in trs_list:
            f.write(f"!TRS_NAME {trs_name}\n")
            f.write("!NB_TRs 1\n")
            f.write(f"!NB_GEOM_ENTS {len(faces)}\n")
            f.write("!GEOM_TYPE Face\n")
            f.write("!LIST_GEOM_ENT\n")
            for n0, n1 in faces:
                f.write(f"2 0 {n0} {n1}\n")

        f.write("!LIST_NODE \n")
        for k in range(n_nodes):
            f.write(f"{nodes[k, 0]:.15e} {nodes[k, 1]:.15e}\n")

        f.write("!LIST_STATE 1\n")
        for s in range(n_states):
            vals = " ".join(f"{states[s, eq]:.16e}" for eq in range(nb_eq))
            f.write(f"{vals}\n")

        f.write("!END\n")

    print(f"  Wrote: {path}")
    print(f"  {n_nodes} nodes, {n_cells} elements, {n_states} states (P1)")
    for name, faces in trs_list:
        print(f"    TRS {name}: {len(faces)} faces")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Direct P1 interpolation: .msh + old P1 -> new P1 CFmesh + R-H file")
    parser.add_argument('--msh', required=True,
                        help='Split Gmsh .msh file (from split_at_b23.py)')
    parser.add_argument('--old-solution', required=True,
                        help='Converged P1 CFmesh (SOL_POLYORDER must be 1)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for mesh_clean.CFmesh and RH_postshock.dat')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ----------------------------------------------------------
    # 1. Parse new .msh
    # ----------------------------------------------------------
    print(f"Parsing new mesh: {args.msh}")
    new_nodes, new_quads, trs_faces = parse_split_msh(args.msh)
    n_cells = len(new_quads)
    print(f"  {len(new_nodes)} nodes, {n_cells} elements")
    for name, faces in sorted(trs_faces.items()):
        print(f"    TRS {name}: {len(faces)} faces")

    # ----------------------------------------------------------
    # 2. Parse old P1 CFmesh
    # ----------------------------------------------------------
    print(f"\nParsing old solution: {args.old_solution}")
    old_data = parse_cfmesh_with_validation(args.old_solution, required_polyorder=1)
    nb_eq = old_data['nb_eq']
    print(f"  {old_data['nb_elem']} elements, {old_data['nb_states']} states, "
          f"SOL_POLYORDER={old_data['sol_polyorder']} (P1), {nb_eq} equations")

    # ----------------------------------------------------------
    # 3. Compute GL coordinates for old and new meshes
    # ----------------------------------------------------------
    print("\nComputing GL solution point coordinates...")
    old_coords = compute_state_coordinates(old_data)

    new_n_states = 4 * n_cells
    new_coords = compute_gl_coordinates(new_nodes, new_quads)
    print(f"  Old mesh: {len(old_coords)} states")
    print(f"  New mesh: {new_n_states} states")

    # ----------------------------------------------------------
    # 4. Identify upstream/downstream domains
    # ----------------------------------------------------------
    print("\nIdentifying upstream domain...")
    upstream_elems = identify_upstream_elements(new_quads, trs_faces)
    upstream_states = set()
    for ci in upstream_elems:
        for iSol in range(4):
            upstream_states.add(4 * ci + iSol)
    downstream_states = set(range(new_n_states)) - upstream_states
    print(f"  Upstream: {len(upstream_elems)} elements, {len(upstream_states)} states")
    print(f"  Downstream: {n_cells - len(upstream_elems)} elements, {len(downstream_states)} states")

    # Old mesh upstream/downstream
    old_upstream, _ = identify_upstream_domain(old_data)
    old_dn_ids = sorted(set(range(old_data['nb_states'])) - old_upstream)
    old_dn_coords = old_coords[old_dn_ids]
    old_dn_states = old_data['states'][old_dn_ids]
    print(f"  Old mesh: {len(old_dn_ids)} downstream, {len(old_upstream)} upstream states")

    # ----------------------------------------------------------
    # 5. Interpolate: old P1 downstream -> new P1 downstream
    # ----------------------------------------------------------
    print("\nInterpolating old P1 solution onto new mesh (KD-tree nearest-state)...")
    new_states = np.zeros((new_n_states, nb_eq))

    # Set ALL states to freestream first
    for s in range(new_n_states):
        new_states[s] = FREESTREAM

    # Overwrite downstream states with interpolated values
    new_dn_ids = sorted(downstream_states)
    new_dn_coords = new_coords[new_dn_ids]

    tree = cKDTree(old_dn_coords)
    distances, indices = tree.query(new_dn_coords)

    for i, new_sid in enumerate(new_dn_ids):
        new_states[new_sid] = old_dn_states[indices[i]]

    print(f"  Interpolated {len(new_dn_ids)} downstream states")
    print(f"  Max distance: {distances.max()*1000:.4f} mm")
    print(f"  Mean distance: {distances.mean()*1000:.4f} mm")
    print(f"  Upstream: {len(upstream_states)} states set to freestream")

    # ----------------------------------------------------------
    # 6. Write P1 CFmesh
    # ----------------------------------------------------------
    clean_path = os.path.join(args.output_dir, 'mesh_clean.CFmesh')
    print(f"\nWriting P1 CFmesh...")
    write_p1_cfmesh(clean_path, new_nodes, new_quads, trs_faces,
                    new_states, nb_eq)

    # ----------------------------------------------------------
    # 7. Compute R-H post-shock state (PLATO)
    # ----------------------------------------------------------
    # Re-parse the written CFmesh for R-H computation
    # (reuses the existing compute_rh_all_faces which expects parsed CFmesh data)
    print("\n--- Computing R-H post-shock state (PLATO) ---")
    rh_data = parse_cfmesh_with_validation(clean_path, required_polyorder=1)
    plato_init()
    rh_results = compute_rh_all_faces(rh_data)
    rh_path = os.path.join(args.output_dir, 'RH_postshock.dat')
    write_rh_file(rh_results, rh_path)
    plato_finalize()

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"DONE (direct P1 interpolation, no MeshUpgrade needed)")
    print(f"{'='*60}")
    print(f"  P1 CFmesh:     {clean_path}")
    print(f"  R-H BC file:   {rh_path}")
    print(f"  Downstream:    from {args.old_solution} (P1 nearest-state)")
    print(f"  Upstream:      freestream")
    print(f"\n  Update CFcase:")
    print(f"    CFmeshFileReader.Data.FileName = {clean_path}")
    print(f"    Data.SShockDown.InputFileName = {rh_path}")


if __name__ == '__main__':
    main()
