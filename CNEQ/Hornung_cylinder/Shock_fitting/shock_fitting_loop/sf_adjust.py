#!/usr/bin/env python3
"""
SHOCK FITTING — ADJUST SHOCK POSITION (from converged P1 solution)
===================================================================

For SUBSEQUENT shock fitting iterations. Takes a new P1 mesh (with
adjusted shock position) and initializes it from a previous converged
P1 solution via nearest-state interpolation.

Input:  New P1 mesh + old converged P1 solution
Output: mesh_clean.CFmesh + RH_postshock.dat

Validation:
  - ERRORS if new mesh is P0 — you forgot MeshUpgrade
  - ERRORS if old solution is P0 — must be a converged P1 solution
  - ERRORS if --old-solution is not given — use sf_init.py instead

Usage:
  LD_LIBRARY_PATH=.../plato.../lib:$LD_LIBRARY_PATH \\
  python3 sf_adjust.py \\
    --p1-mesh <new_P1_from_MeshUpgrade.CFmesh> \\
    --old-solution <converged_P1.CFmesh> \\
    --output-dir <dir>

What it does:
  1. Validates: new mesh MUST be P1, old solution MUST be P1
  2. Interpolates old P1 downstream solution → new mesh (nearest-state, KD-tree)
  3. Sets upstream domain to freestream
  4. Computes R-H post-shock state using PLATO at each ShockDown face
  5. Writes cleaned mesh + R-H file
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _sf_common import (
    parse_cfmesh_with_validation, identify_upstream_domain,
    set_upstream_to_freestream, write_cfmesh_states,
    interpolate_p1_solution,
    compute_rh_all_faces, write_rh_file,
    plato_init, plato_finalize,
    FREESTREAM, get_freestream_thermo
)


def main():
    parser = argparse.ArgumentParser(
        description="SF Adjust: prepare new mesh with solution carryover from old P1")
    parser.add_argument('--p1-mesh', required=True,
                        help='New P1 CFmesh from MeshUpgrade (SOL_POLYORDER must be 1)')
    parser.add_argument('--old-solution', required=True,
                        help='Previous converged P1 CFmesh (SOL_POLYORDER must be 1)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for mesh_clean.CFmesh and RH_postshock.dat')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse and validate NEW mesh: MUST be P1
    print(f"Reading new mesh: {args.p1_mesh}")
    data = parse_cfmesh_with_validation(args.p1_mesh, required_polyorder=1)
    print(f"  {data['nb_elem']} elements, {data['nb_states']} states, "
          f"SOL_POLYORDER={data['sol_polyorder']} (P1) ✓")

    # Parse and validate OLD solution: MUST be P1
    print(f"\nReading old solution: {args.old_solution}")
    old_data = parse_cfmesh_with_validation(args.old_solution, required_polyorder=1)
    print(f"  {old_data['nb_elem']} elements, {old_data['nb_states']} states, "
          f"SOL_POLYORDER={old_data['sol_polyorder']} (P1) ✓")

    # Identify upstream domains
    upstream_states_new, upstream_elems_new = identify_upstream_domain(data)
    print(f"\n  New mesh upstream: {len(upstream_elems_new)} elements, "
          f"{len(upstream_states_new)} states")

    # Interpolate old P1 solution onto new mesh (downstream only)
    print(f"\n--- Interpolating old P1 solution onto new mesh ---")
    interpolate_p1_solution(data, old_data, upstream_states_new)

    # Set upstream to freestream
    print("\n--- Setting upstream to freestream ---")
    set_upstream_to_freestream(data, upstream_states_new)

    # Write cleaned mesh
    clean_path = os.path.join(args.output_dir, 'mesh_clean.CFmesh')
    write_cfmesh_states(data, clean_path)

    # Compute R-H with PLATO
    print("\n--- Computing R-H post-shock state (PLATO) ---")
    plato_init()
    rh_results = compute_rh_all_faces(data)
    rh_path = os.path.join(args.output_dir, 'RH_postshock.dat')
    write_rh_file(rh_results, rh_path)
    plato_finalize()

    print(f"\n{'='*60}")
    print(f"DONE (adjusted shock, P1 solution carryover)")
    print(f"{'='*60}")
    print(f"  Cleaned mesh:  {clean_path}")
    print(f"  R-H BC file:   {rh_path}")
    print(f"  Downstream:    from {args.old_solution} (P1 interpolation)")
    print(f"  Upstream:      freestream")
    print(f"\n  Update CFcase:")
    print(f"    CFmeshFileReader.Data.FileName = {clean_path}")
    print(f"    Data.SShockDown.InputFileName = {rh_path}")


if __name__ == '__main__':
    main()
