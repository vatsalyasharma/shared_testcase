#!/usr/bin/env python3
"""
SHOCK FITTING — INITIAL SETUP (first iteration, no prior P1 solution)
=====================================================================

For the FIRST shock fitting iteration only. Takes a P1 mesh
(upgraded from P0 via MeshUpgrade) and prepares it for simulation.
Downstream states are kept from the P0→P1 projection already in the CFmesh.
Upstream states are set to freestream.

Input:  P1 CFmesh (SOL_POLYORDER=1, from MeshUpgrade)
Output: mesh_clean.CFmesh + RH_postshock.dat

Validation:
  - ERRORS if input is P0 (SOL_POLYORDER=0) — you forgot MeshUpgrade
  - For subsequent iterations with P1 solution carryover, use sf_adjust.py

Usage:
  LD_LIBRARY_PATH=.../plato.../lib:$LD_LIBRARY_PATH \\
  python3 sf_init.py --p1-mesh <P1_from_MeshUpgrade.CFmesh> --output-dir <dir>

What it does:
  1. Validates: input MUST be P1 (SOL_POLYORDER=1)
  2. Sets upstream domain to freestream (fixes interpolation contamination)
  3. Keeps downstream states from P0 interpolation (already in CFmesh)
  4. Computes R-H post-shock state using PLATO at each ShockDown face
  5. Writes cleaned mesh + R-H file
"""

import argparse
import sys
import os

# Shared code lives in _sf_common.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _sf_common import (
    parse_cfmesh_with_validation, identify_upstream_domain,
    set_upstream_to_freestream, write_cfmesh_states,
    compute_rh_all_faces, write_rh_file,
    plato_init, plato_finalize,
    FREESTREAM, get_freestream_thermo
)


def main():
    parser = argparse.ArgumentParser(
        description="SF Init: prepare P1 mesh for first shock fitting run (no prior solution)")
    parser.add_argument('--p1-mesh', required=True,
                        help='P1 CFmesh from MeshUpgrade (SOL_POLYORDER must be 1)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for mesh_clean.CFmesh and RH_postshock.dat')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse and validate: MUST be P1
    print(f"Reading: {args.p1_mesh}")
    data = parse_cfmesh_with_validation(args.p1_mesh, required_polyorder=1)
    print(f"  {data['nb_elem']} elements, {data['nb_states']} states, "
          f"SOL_POLYORDER={data['sol_polyorder']} (P1) ✓")

    # Identify upstream domain
    upstream_states, upstream_elems = identify_upstream_domain(data)
    print(f"  Upstream: {len(upstream_elems)} elements, {len(upstream_states)} states")

    # Set upstream to freestream
    print("\n--- Setting upstream to freestream ---")
    set_upstream_to_freestream(data, upstream_states)

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
    print(f"DONE (initial setup, no prior solution)")
    print(f"{'='*60}")
    print(f"  Cleaned mesh:  {clean_path}")
    print(f"  R-H BC file:   {rh_path}")
    print(f"  Downstream:    from P0→P1 projection (already in CFmesh)")
    print(f"  Upstream:      freestream")
    print(f"\n  Update CFcase:")
    print(f"    CFmeshFileReader.Data.FileName = {clean_path}")
    print(f"    Data.SShockDown.InputFileName = {rh_path}")


if __name__ == '__main__':
    main()
