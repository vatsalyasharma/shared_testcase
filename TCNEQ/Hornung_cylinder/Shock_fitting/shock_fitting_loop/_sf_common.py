"""
Shared code for shock fitting loop scripts.
Validation, PLATO interface, R-H solver, interpolation, I/O.
"""

import ctypes
import numpy as np
from scipy.spatial import cKDTree
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
from fix_upstream_and_compute_rh_v2 import (
    parse_cfmesh as _parse_cfmesh_raw,
    compute_face_normals,
    identify_upstream_domain,
)

# ============================================================
# Constants
# ============================================================

FREESTREAM = [0.0001952, 0.004956, 5590.0, 0.0, 1833.0]  # rho_N, rho_N2, u, v, T

PLATO_LIB_PATH = '/mnt/c/Research/codes/Solvers/PLATO/plato_install/lib/libplato.so'
PLATO_DB_PATH = '/mnt/c/Research/codes/Solvers/PLATO/plato_database-main_feb26/database-main'


# ============================================================
# CFmesh parsing with SOL_POLYORDER validation
# ============================================================

def parse_cfmesh_with_validation(filepath, required_polyorder=None):
    """Parse CFmesh and validate SOL_POLYORDER if required."""
    data = _parse_cfmesh_raw(filepath)

    # Extract SOL_POLYORDER from raw lines
    sol_polyorder = None
    for line in data['lines']:
        s = line.strip()
        if s.startswith('!SOL_POLYORDER'):
            sol_polyorder = int(s.split()[1])
            break

    if sol_polyorder is None:
        print(f"WARNING: SOL_POLYORDER not found in {filepath}")
        sol_polyorder = -1

    data['sol_polyorder'] = sol_polyorder

    if required_polyorder is not None and sol_polyorder != required_polyorder:
        poly_name = {0: 'P0', 1: 'P1', 2: 'P2'}.get(sol_polyorder, f'P{sol_polyorder}')
        req_name = {0: 'P0', 1: 'P1', 2: 'P2'}.get(required_polyorder, f'P{required_polyorder}')
        print(f"\n{'='*60}")
        print(f"ERROR: Expected {req_name} mesh (SOL_POLYORDER={required_polyorder})")
        print(f"       Got {poly_name} mesh (SOL_POLYORDER={sol_polyorder})")
        print(f"       File: {filepath}")
        if required_polyorder == 1 and sol_polyorder == 0:
            print(f"\n  Did you forget to run MeshUpgrade?")
            print(f"  Run: ./coolfluid-solver --scase MeshUpgrade_P1.CFcase")
        elif required_polyorder == 0 and sol_polyorder == 1:
            print(f"\n  This is a P1 mesh. For initial setup use sf_init.py with a P1 mesh.")
            print(f"  For shock adjustment use sf_adjust.py with --old-solution.")
        print(f"{'='*60}")
        sys.exit(1)

    return data


# ============================================================
# Upstream domain handling
# ============================================================

def set_upstream_to_freestream(data, upstream_states):
    """Set all upstream states to freestream. Reports contamination."""
    states = data['states']
    n_up = len(upstream_states)
    contaminated = sum(1 for sid in upstream_states if states[sid, 4] > 2500)
    print(f"  Upstream states: {n_up}")
    print(f"  Contaminated (T>2500K): {contaminated}")
    for sid in upstream_states:
        for eq in range(data['nb_eq']):
            states[sid, eq] = FREESTREAM[eq]
    print(f"  All upstream states set to freestream ✓")


def write_cfmesh_states(data, output_path):
    """Write CFmesh with updated state values."""
    lines = list(data['lines'])
    st_start = data['states_line_start']
    states = data['states']
    for s in range(data['nb_states']):
        vals = ' '.join(f'{states[s, eq]:.16e}' for eq in range(data['nb_eq']))
        lines[st_start + s] = vals + '\n'
    with open(output_path, 'w') as f:
        f.writelines(lines)
    print(f"  Wrote: {output_path}")


# ============================================================
# P1 solution interpolation (nearest-state, KD-tree)
# ============================================================

def compute_state_coordinates(data):
    """
    Compute (x,y) of each solution point via bilinear mapping.
    P1 quad: 4 GL points at xi,eta = +/-1/sqrt(3).

    LIMITATION (Fix #5): Hardcoded for P1 quads with GaussLegendre distribution.
    Will silently give wrong coordinates for P2 (9 points), triangles,
    or non-GL point distributions. SOL_POLYORDER is validated by the caller
    but element type and point distribution are NOT checked.
    """
    # Validate what we can
    if data.get('sol_polyorder', -1) != 1:
        print(f"WARNING: compute_state_coordinates assumes P1 (SOL_POLYORDER=1), "
              f"got {data.get('sol_polyorder', '?')}")

    # Check states per element (P1 quad = 4)
    if data['nb_states'] > 0 and data['nb_elem'] > 0:
        spe = data['nb_states'] / data['nb_elem']
        if abs(spe - 4.0) > 0.01:  # not exactly 4 states per element
            print(f"WARNING: Expected 4 states/element (P1 quad), "
                  f"got {spe:.1f}. Coordinates may be wrong.")
    nodes = data['nodes']
    elems = data['elems']
    nb_states = data['nb_states']

    g = 1.0 / np.sqrt(3.0)
    gl_ref = [(-g, -g), (g, -g), (-g, g), (g, g)]

    coords = np.zeros((nb_states, 2))
    for elem in elems:
        enodes = elem[:4]
        estates = elem[4:]
        x = np.array([nodes[n][0] for n in enodes])
        y = np.array([nodes[n][1] for n in enodes])
        for iSol, sid in enumerate(estates):
            if iSol >= len(gl_ref):
                break
            xi, eta = gl_ref[iSol]
            N0 = 0.25 * (1 - xi) * (1 - eta)
            N1 = 0.25 * (1 + xi) * (1 - eta)
            N2 = 0.25 * (1 + xi) * (1 + eta)
            N3 = 0.25 * (1 - xi) * (1 + eta)
            coords[sid, 0] = N0*x[0] + N1*x[1] + N2*x[2] + N3*x[3]
            coords[sid, 1] = N0*y[0] + N1*y[1] + N2*y[2] + N3*y[3]
    return coords


def interpolate_p1_solution(new_data, old_data, new_upstream_states):
    """
    Interpolate old P1 downstream solution onto new mesh.
    Uses nearest-state mapping (KD-tree) in physical coordinates.
    Only downstream states are interpolated.
    """
    print("  Computing state coordinates (old mesh)...")
    old_coords = compute_state_coordinates(old_data)

    print("  Computing state coordinates (new mesh)...")
    new_coords = compute_state_coordinates(new_data)

    # Old downstream states only
    old_upstream, _ = identify_upstream_domain(old_data)
    old_dn_ids = sorted(set(range(old_data['nb_states'])) - old_upstream)
    old_dn_coords = old_coords[old_dn_ids]
    old_dn_states = old_data['states'][old_dn_ids]
    print(f"  Old mesh: {len(old_dn_ids)} downstream, {len(old_upstream)} upstream")

    # New downstream states
    new_dn_ids = sorted(set(range(new_data['nb_states'])) - new_upstream_states)
    new_dn_coords = new_coords[new_dn_ids]

    # KD-tree nearest-neighbor lookup
    tree = cKDTree(old_dn_coords)
    distances, indices = tree.query(new_dn_coords)

    # Copy
    for i, new_sid in enumerate(new_dn_ids):
        new_data['states'][new_sid] = old_dn_states[indices[i]]

    print(f"  Interpolated {len(new_dn_ids)} downstream states")
    print(f"  Max distance: {distances.max()*1000:.4f} mm, "
          f"mean: {distances.mean()*1000:.4f} mm")


# ============================================================
# PLATO interface
# ============================================================

_lib = None
_nb_sp = 0
_Ri = None
_dp = ctypes.POINTER(ctypes.c_double)


def plato_init():
    global _lib, _nb_sp, _Ri
    gf = ctypes.CDLL('libgfortran.so.5', mode=ctypes.RTLD_GLOBAL)
    _lib = ctypes.CDLL(PLATO_LIB_PATH)
    _lib.initializeC(
        b'Python', b'nitrogen2', b'nitrogen2', b'empty',
        PLATO_DB_PATH.encode(),
        ctypes.c_int(6), ctypes.c_int(9), ctypes.c_int(9),
        ctypes.c_int(5), ctypes.c_int(len(PLATO_DB_PATH)))
    _lib.get_nb_species.restype = ctypes.c_int
    _nb_sp = _lib.get_nb_species()
    _Ri = np.zeros(_nb_sp, dtype=np.float64)
    _lib.get_Ri(_Ri.ctypes.data_as(_dp))


def plato_finalize():
    global _lib
    if _lib is not None:
        _lib.finalize()
        _lib = None


def _plato_mixture_h(T, Y_N, Y_N2):
    temp = np.array([float(T)], dtype=np.float64)
    hi = np.zeros(_nb_sp, dtype=np.float64)
    _lib.species_enthalpy(temp.ctypes.data_as(_dp), hi.ctypes.data_as(_dp))
    return Y_N * hi[0] + Y_N2 * hi[1]


def get_freestream_thermo():
    """Return (R_mix, gamma_guess, a_guess, p_fs) from freestream + PLATO Ri."""
    rho_N_fs, rho_N2_fs, u_fs, v_fs, T_fs = FREESTREAM
    rho_fs = rho_N_fs + rho_N2_fs
    Y_N = rho_N_fs / rho_fs
    Y_N2 = rho_N2_fs / rho_fs
    R_mix = Y_N * _Ri[0] + Y_N2 * _Ri[1]
    p_fs = rho_fs * R_mix * T_fs
    # RRHO gamma for initial guess
    theta_v = 3395.0
    x = theta_v / T_fs
    cp_vib = _Ri[1] * x**2 * np.exp(x) / (np.exp(x) - 1)**2
    cp_mix = Y_N * _Ri[0] * 2.5 + Y_N2 * (_Ri[1] * 3.5 + cp_vib)
    gamma_guess = cp_mix / (cp_mix - R_mix)
    a_guess = np.sqrt(gamma_guess * R_mix * T_fs)
    return R_mix, gamma_guess, a_guess, p_fs


# ============================================================
# R-H solver
# ============================================================

def _solve_rh_single(rho_fs, u_n_fs, p_fs, T_fs, Y_N, Y_N2,
                     R_mix, gamma_guess, a_guess):
    """Solve R-H normal shock (Vs=0) at one face using PLATO enthalpies."""
    m = rho_fs * u_n_fs
    h1 = _plato_mixture_h(T_fs, Y_N, Y_N2)
    M_n = abs(u_n_fs) / a_guess
    gp1 = gamma_guess + 1
    gm1 = gamma_guess - 1
    rho2 = rho_fs * gp1 * M_n**2 / (gm1 * M_n**2 + 2)

    for it in range(200):
        u_n2 = m / rho2
        p2 = p_fs + m * (u_n_fs - u_n2)
        h2_target = h1 + 0.5 * (u_n_fs**2 - u_n2**2)
        T2 = max(p2 / (rho2 * R_mix), 100.0)
        h2_actual = _plato_mixture_h(T2, Y_N, Y_N2)
        res = h2_actual - h2_target
        if abs(res) < 1e-10 * abs(h2_target):
            return rho2, u_n2, p2, T2, it
        drho = max(abs(rho2) * 1e-6, 1e-12)
        rho2p = rho2 + drho
        u_n2p = m / rho2p
        p2p = p_fs + m * (u_n_fs - u_n2p)
        h2_tp = h1 + 0.5 * (u_n_fs**2 - u_n2p**2)
        T2p = max(p2p / (rho2p * R_mix), 100.0)
        res_p = _plato_mixture_h(T2p, Y_N, Y_N2) - h2_tp
        dres = (res_p - res) / drho
        if abs(dres) < 1e-30:
            break
        rho2 -= 0.8 * res / dres
        rho2 = max(rho2, rho_fs * 1.01)
    return rho2, u_n2, p2, T2, it


def compute_rh_all_faces(data):
    """Compute R-H post-shock state at all ShockDown faces using PLATO."""
    rho_N_fs, rho_N2_fs, u_fs, v_fs, T_fs = FREESTREAM
    rho_fs = rho_N_fs + rho_N2_fs
    Y_N = rho_N_fs / rho_fs
    Y_N2 = rho_N2_fs / rho_fs
    R_mix, gamma_guess, a_guess, p_fs = get_freestream_thermo()

    print(f"  Freestream: rho={rho_fs:.5f}, u={u_fs:.0f}, T={T_fs:.0f}")
    print(f"  R_mix={R_mix:.2f}, gamma={gamma_guess:.4f}, a={a_guess:.0f}")

    face_data = compute_face_normals(data)
    results = []

    print(f"  {'idx':>3} {'y':>8} {'M_n':>6} {'T':>7} {'u_n':>8} {'rho_r':>6} {'it':>3}")
    for fd in face_data:
        nx, ny = fd['normal']
        u_n_fs_loc = u_fs * nx + v_fs * ny
        u_t_fs_loc = -u_fs * ny + v_fs * nx
        M_n = abs(u_n_fs_loc) / a_guess

        rho2, u_n2, p2, T2, n_it = _solve_rh_single(
            rho_fs, u_n_fs_loc, p_fs, T_fs, Y_N, Y_N2,
            R_mix, gamma_guess, a_guess)

        u_post = u_n2 * nx - u_t_fs_loc * ny
        v_post = u_n2 * ny + u_t_fs_loc * nx

        y_c = fd['center'][1]
        print(f"  {fd['face_idx']:3d} {y_c:8.4f} {M_n:6.2f} {T2:7.0f} "
              f"{u_n2:8.1f} {rho2/rho_fs:6.2f} {n_it:3d}")

        results.append({
            'y': y_c,
            'rho_N': Y_N * rho2, 'rho_N2': Y_N2 * rho2,
            'u': u_post, 'v': v_post, 'T': T2,
        })

    stag = [r for r in results if abs(r['y']) < 0.001]
    if stag:
        s = stag[0]
        print(f"  Stagnation: T={s['T']:.0f}K, u={s['u']:.1f}")

    return results


def write_rh_file(results, output_path):
    """Write R-H post-shock state in BCDirichletFromFile format."""
    with open(output_path, 'w') as f:
        f.write("y rho_N rho_N2 u v T\n")
        f.write(f"{len(results)}\n")
        for r in results:
            f.write(f"{r['y']:.10e} {r['rho_N']:.10e} {r['rho_N2']:.10e} "
                    f"{r['u']:.10e} {r['v']:.10e} {r['T']:.10e}\n")
    print(f"  Wrote: {output_path}")
