#!/usr/bin/env python3
"""
SHOCK FITTING LOOP — Step 2: Prepare New Mesh for Simulation
=============================================================

After the user has regenerated the mesh in Gmsh and upgraded to P1,
this script:
  1. Initializes the solution on the new mesh:
     - FIRST TIME (no --old-solution): keeps P0 interpolation for downstream,
       sets upstream to freestream
     - SUBSEQUENT (--old-solution given): interpolates old P1 solution onto
       new mesh (nearest-state), sets upstream to freestream
  2. Computes R-H post-shock state using PLATO (exact thermodynamics)
  3. Writes cleaned CFmesh + R-H data file

Usage (first time):
  LD_LIBRARY_PATH=.../plato.../lib:$LD_LIBRARY_PATH \
  python3 step2_prepare_mesh.py --new-p1-mesh <new_P1.CFmesh> --output-dir <dir>

Usage (subsequent iterations — with solution carryover):
  LD_LIBRARY_PATH=.../plato.../lib:$LD_LIBRARY_PATH \
  python3 step2_prepare_mesh.py --new-p1-mesh <new_P1.CFmesh> \
    --old-solution <old_converged.CFmesh> --output-dir <dir>

Outputs:
  <dir>/mesh_clean.CFmesh       — P1 mesh with initialized solution
  <dir>/RH_postshock.dat        — R-H post-shock state for DirichletFromFile BC
"""

import argparse
import ctypes
import numpy as np
from scipy.spatial import cKDTree
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
from fix_upstream_and_compute_rh_v2 import (
    parse_cfmesh, compute_face_normals, identify_upstream_domain
)

# ============================================================
# PLATO interface
# ============================================================

PLATO_LIB_PATH = '/mnt/c/Research/codes/Solvers/PLATO/plato_install/lib/libplato.so'
PLATO_DB_PATH = '/mnt/c/Research/codes/Solvers/PLATO/plato_database-main_feb26/database-main'

_plato_lib = None
_plato_nb_sp = 0
_plato_Ri = None
_dp = ctypes.POINTER(ctypes.c_double)


def plato_init():
    global _plato_lib, _plato_nb_sp, _plato_Ri, _dp
    gf = ctypes.CDLL('libgfortran.so.5', mode=ctypes.RTLD_GLOBAL)
    _plato_lib = ctypes.CDLL(PLATO_LIB_PATH)
    _dp = ctypes.POINTER(ctypes.c_double)
    _plato_lib.initializeC(
        b'Python', b'nitrogen2', b'nitrogen2', b'empty',
        PLATO_DB_PATH.encode(),
        ctypes.c_int(6), ctypes.c_int(9), ctypes.c_int(9),
        ctypes.c_int(5), ctypes.c_int(len(PLATO_DB_PATH)))
    _plato_lib.get_nb_species.restype = ctypes.c_int
    _plato_nb_sp = _plato_lib.get_nb_species()
    _plato_Ri = np.zeros(_plato_nb_sp, dtype=np.float64)
    _plato_lib.get_Ri(_plato_Ri.ctypes.data_as(_dp))


def plato_finalize():
    if _plato_lib is not None:
        _plato_lib.finalize()


def plato_mixture_h(T, Y_N, Y_N2):
    temp = np.array([float(T)], dtype=np.float64)
    hi = np.zeros(_plato_nb_sp, dtype=np.float64)
    _plato_lib.species_enthalpy(temp.ctypes.data_as(_dp), hi.ctypes.data_as(_dp))
    return Y_N * hi[0] + Y_N2 * hi[1]


# ============================================================
# R-H solver with PLATO
# ============================================================

def solve_rh_plato(rho_fs, u_n_fs, p_fs, T_fs, Y_N, Y_N2, R_mix, gamma_guess, a_guess):
    """Solve R-H normal shock (Vs=0) using PLATO enthalpies."""
    m = rho_fs * u_n_fs
    h1 = plato_mixture_h(T_fs, Y_N, Y_N2)
    M_n = abs(u_n_fs) / a_guess
    gp1 = gamma_guess + 1
    gm1 = gamma_guess - 1
    rho2 = rho_fs * gp1 * M_n**2 / (gm1 * M_n**2 + 2)

    for it in range(200):
        u_n2 = m / rho2
        p2 = p_fs + m * (u_n_fs - u_n2)
        h2_target = h1 + 0.5 * (u_n_fs**2 - u_n2**2)
        T2 = max(p2 / (rho2 * R_mix), 100.0)
        h2_actual = plato_mixture_h(T2, Y_N, Y_N2)
        res = h2_actual - h2_target
        if abs(res) < 1e-10 * abs(h2_target):
            return rho2, u_n2, p2, T2, it
        drho = max(abs(rho2) * 1e-6, 1e-12)
        rho2p = rho2 + drho
        u_n2p = m / rho2p
        p2p = p_fs + m * (u_n_fs - u_n2p)
        h2_tp = h1 + 0.5 * (u_n_fs**2 - u_n2p**2)
        T2p = max(p2p / (rho2p * R_mix), 100.0)
        res_p = plato_mixture_h(T2p, Y_N, Y_N2) - h2_tp
        dres = (res_p - res) / drho
        if abs(dres) < 1e-30:
            break
        rho2 -= 0.8 * res / dres
        rho2 = max(rho2, rho_fs * 1.01)
    return rho2, u_n2, p2, T2, it


# ============================================================
# Solution interpolation (P1 → new P1 mesh)
# ============================================================

def compute_state_coordinates(data):
    """
    Compute physical (x,y) coordinates of each state (solution point).
    For P1 FR on quads, the 4 GL points are at reference positions
    mapped through the element geometry.

    Approximation: use weighted average of element node coordinates.
    For P1 (2x2 GL points at xi,eta = +/-1/sqrt(3)):
      GL ref coords: (-0.577, -0.577), (0.577, -0.577), (-0.577, 0.577), (0.577, 0.577)
    """
    nodes = data['nodes']
    elems = data['elems']
    nb_eq = data['nb_eq']
    nb_states = data['nb_states']

    # GL reference coordinates for P1 (2x2)
    g = 1.0 / np.sqrt(3.0)
    gl_ref = [(-g, -g), (g, -g), (-g, g), (g, g)]

    state_coords = np.zeros((nb_states, 2))

    for elem in elems:
        enodes = elem[:4]  # corner nodes
        estates = elem[4:]  # states

        # Corner coordinates
        x = np.array([nodes[n][0] for n in enodes])
        y = np.array([nodes[n][1] for n in enodes])

        for iSol, sid in enumerate(estates):
            if iSol >= len(gl_ref):
                break
            xi, eta = gl_ref[iSol]

            # Bilinear shape functions for quad
            N0 = 0.25 * (1 - xi) * (1 - eta)
            N1 = 0.25 * (1 + xi) * (1 - eta)
            N2 = 0.25 * (1 + xi) * (1 + eta)
            N3 = 0.25 * (1 - xi) * (1 + eta)

            state_coords[sid, 0] = N0*x[0] + N1*x[1] + N2*x[2] + N3*x[3]
            state_coords[sid, 1] = N0*y[0] + N1*y[1] + N2*y[2] + N3*y[3]

    return state_coords


def interpolate_old_solution(new_data, old_data, upstream_states):
    """
    Interpolate old P1 solution onto new P1 mesh using nearest-state mapping.
    Only downstream states are interpolated; upstream is set separately to freestream.
    """
    print("  Computing state coordinates for old mesh...")
    old_coords = compute_state_coordinates(old_data)

    print("  Computing state coordinates for new mesh...")
    new_coords = compute_state_coordinates(new_data)

    # Identify old downstream states (exclude upstream)
    old_upstream, _ = identify_upstream_domain(old_data)
    old_downstream_ids = sorted(set(range(old_data['nb_states'])) - old_upstream)
    old_downstream_coords = old_coords[old_downstream_ids]
    old_downstream_states = old_data['states'][old_downstream_ids]

    print(f"  Old mesh: {len(old_downstream_ids)} downstream states, {len(old_upstream)} upstream states")

    # Build KD-tree from old downstream state coordinates
    tree = cKDTree(old_downstream_coords)

    # For each new downstream state, find nearest old downstream state
    new_downstream_ids = sorted(set(range(new_data['nb_states'])) - upstream_states)
    new_downstream_coords = new_coords[new_downstream_ids]

    distances, indices = tree.query(new_downstream_coords)

    # Copy state values
    new_states = new_data['states']
    for i, new_sid in enumerate(new_downstream_ids):
        old_idx = indices[i]
        new_states[new_sid] = old_downstream_states[old_idx]

    max_dist = distances.max()
    mean_dist = distances.mean()
    print(f"  Interpolated {len(new_downstream_ids)} downstream states")
    print(f"  Max distance: {max_dist*1000:.4f} mm, mean: {mean_dist*1000:.4f} mm")

    return new_states


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Step 2: Prepare new mesh for simulation")
    parser.add_argument('--new-p1-mesh', required=True,
                        help='New P1 CFmesh (from MeshUpgrade)')
    parser.add_argument('--old-solution', default=None,
                        help='Previous converged P1 CFmesh for solution carryover. '
                             'If not given, keeps P0 interpolation for downstream.')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Freestream
    freestream = [0.0001952, 0.004956, 5590.0, 0.0, 1833.0]
    rho_N_fs, rho_N2_fs, u_fs, v_fs, T_fs = freestream
    rho_fs = rho_N_fs + rho_N2_fs
    Y_N = rho_N_fs / rho_fs
    Y_N2 = rho_N2_fs / rho_fs

    # Initialize PLATO
    print("Initializing PLATO...")
    plato_init()
    R_mix = Y_N * _plato_Ri[0] + Y_N2 * _plato_Ri[1]
    p_fs = rho_fs * R_mix * T_fs

    # RRHO gamma for initial guess
    theta_v = 3395.0
    R_N2 = _plato_Ri[1]
    x = theta_v / T_fs
    cp_vib = R_N2 * x**2 * np.exp(x) / (np.exp(x) - 1)**2
    cp_mix = Y_N * _plato_Ri[0] * 2.5 + Y_N2 * (R_N2 * 3.5 + cp_vib)
    gamma_guess = cp_mix / (cp_mix - R_mix)
    a_guess = np.sqrt(gamma_guess * R_mix * T_fs)
    print(f"  R_mix={R_mix:.2f}, gamma_guess={gamma_guess:.4f}, a_guess={a_guess:.0f}")

    # Parse new mesh
    print(f"\nReading new mesh: {args.new_p1_mesh}")
    data = parse_cfmesh(args.new_p1_mesh)
    print(f"  {data['nb_elem']} elements, {data['nb_states']} states")

    # Identify upstream domain
    upstream_states, upstream_elems = identify_upstream_domain(data)
    n_up = len(upstream_states)
    print(f"  Upstream: {len(upstream_elems)} elements, {n_up} states")

    # ---- Initialize solution ----
    states = data['states']

    if args.old_solution:
        # SOLUTION CARRYOVER: interpolate old P1 solution onto new mesh
        print(f"\n--- Solution carryover from: {args.old_solution} ---")
        old_data = parse_cfmesh(args.old_solution)
        print(f"  Old mesh: {old_data['nb_elem']} elements, {old_data['nb_states']} states")
        interpolate_old_solution(data, old_data, upstream_states)
    else:
        # FIRST TIME: keep P0 interpolation for downstream (already in the CFmesh)
        print("\n--- First-time initialization (keeping P0 for downstream) ---")
        downstream_T = states[sorted(set(range(data['nb_states'])) - upstream_states), 4]
        print(f"  Downstream T range: {downstream_T.min():.0f} - {downstream_T.max():.0f} K")

    # Always set upstream to freestream
    print("\n--- Setting upstream domain to freestream ---")
    contaminated = sum(1 for sid in upstream_states if states[sid, 4] > 2500)
    print(f"  Contaminated before fix (T>2500K): {contaminated}/{n_up}")
    for sid in upstream_states:
        for eq in range(data['nb_eq']):
            states[sid, eq] = freestream[eq]

    # Write cleaned CFmesh
    clean_mesh_path = os.path.join(args.output_dir, 'mesh_clean.CFmesh')
    lines = list(data['lines'])
    st_start = data['states_line_start']
    for s in range(data['nb_states']):
        vals = ' '.join(f'{states[s, eq]:.16e}' for eq in range(data['nb_eq']))
        lines[st_start + s] = vals + '\n'
    with open(clean_mesh_path, 'w') as f:
        f.writelines(lines)
    print(f"  Wrote: {clean_mesh_path}")

    # ---- Compute R-H at new ShockDown faces ----
    print("\n--- Computing R-H post-shock state (PLATO) ---")
    face_data = compute_face_normals(data)

    rh_results = []
    print(f"{'idx':>3} {'y':>8} {'M_n':>6} {'T':>7} {'u_n':>8} {'rho_r':>6} {'it':>3}")
    for fd in face_data:
        nx, ny = fd['normal']
        u_n_fs_loc = u_fs * nx + v_fs * ny
        u_t_fs_loc = -u_fs * ny + v_fs * nx
        M_n = abs(u_n_fs_loc) / a_guess

        rho2, u_n2, p2, T2, n_it = solve_rh_plato(
            rho_fs, u_n_fs_loc, p_fs, T_fs, Y_N, Y_N2, R_mix, gamma_guess, a_guess)

        u_post = u_n2 * nx - u_t_fs_loc * ny
        v_post = u_n2 * ny + u_t_fs_loc * nx

        y_c = fd['center'][1]
        print(f"{fd['face_idx']:3d} {y_c:8.4f} {M_n:6.2f} {T2:7.0f} {u_n2:8.1f} "
              f"{rho2/rho_fs:6.2f} {n_it:3d}")

        rh_results.append({
            'y': y_c,
            'rho_N': Y_N * rho2, 'rho_N2': Y_N2 * rho2,
            'u': u_post, 'v': v_post, 'T': T2,
        })

    rh_path = os.path.join(args.output_dir, 'RH_postshock.dat')
    with open(rh_path, 'w') as f:
        f.write("y rho_N rho_N2 u v T\n")
        f.write(f"{len(rh_results)}\n")
        for r in rh_results:
            f.write(f"{r['y']:.10e} {r['rho_N']:.10e} {r['rho_N2']:.10e} "
                    f"{r['u']:.10e} {r['v']:.10e} {r['T']:.10e}\n")
    print(f"\n  Wrote: {rh_path}")

    plato_finalize()

    stag = [r for r in rh_results if abs(r['y']) < 0.001]
    if stag:
        s = stag[0]
        print(f"\n  Stagnation: T={s['T']:.0f}K, u={s['u']:.1f}, v={s['v']:.1f}")

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"  Cleaned mesh:  {clean_mesh_path}")
    print(f"  R-H BC file:   {rh_path}")
    if args.old_solution:
        print(f"  Init from:     {args.old_solution} (P1 solution carryover)")
    else:
        print(f"  Init from:     P0 interpolation (first time)")
    print(f"  Upstream:      freestream")
    print(f"\n  Update CFcase:")
    print(f"    CFmeshFileReader.Data.FileName = {os.path.basename(clean_mesh_path)}")
    print(f"    Data.SShockDown.InputFileName = {os.path.basename(rh_path)}")


if __name__ == '__main__':
    main()
