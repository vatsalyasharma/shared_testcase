#!/usr/bin/env python3
"""
Shock-Fitted Mesh Initialization Fixer + Real-Gas R-H Post-Shock State
=======================================================================

Step 1: Fix upstream domain initialization
  - Identifies upstream sub-domain (connected to ShockUp TRS)
  - Sets all upstream states to freestream
  - Keeps downstream states from P0 interpolation
  - Writes cleaned CFmesh

Step 2: Compute Rankine-Hugoniot post-shock state (REAL GAS, no frozen gamma)
  - Uses NASA-7 polynomials for N2 and N thermodynamics
  - Iterative R-H solver (Newton on rho_post)
  - Frozen composition across shock (Y_s preserved)
  - Writes data file for BCDirichletFromFile

Author: COOLFluiD Shock Fitting Extension
Date: 2026-03-18
"""

import argparse
import numpy as np
import os
import sys


# ============================================================
# Thermodynamic Model: Rigid-Rotor Harmonic-Oscillator (RRHO)
# Valid at ALL temperatures (no polynomial extrapolation issues)
# ============================================================

# Molar masses in kg/mol
M_N2 = 0.0280134
M_N  = 0.0140067

# Species gas constants in J/(kg K)
R_UNIV = 8.31446
R_N2 = R_UNIV / M_N2  # 296.803
R_N  = R_UNIV / M_N   # 593.606

# Characteristic vibrational temperature of N2
THETA_V_N2 = 3395.0  # K

# Formation enthalpy of N atom (from N2 dissociation)
# D_0(N2) = 9.759 eV = 941.3 kJ/mol of N2 → per kg of N atoms:
# h_f_N = D_0 * N_A / (2 * M_N) = 9.413e5 / (2 * 0.014) = 3.362e7 J/kg
H_FORM_N = 3.3618e7  # J/kg


def species_h(species, T):
    """
    Species enthalpy in J/kg (includes formation enthalpy).
    RRHO model: trans + rot + vib (harmonic) + formation.
    Valid for all T > 0.
    """
    if species == 'N2':
        # Diatomic: h = (7/2)*R*T + R*theta_v/(exp(theta_v/T) - 1)
        # 7/2 = 5/2 (trans) + 1 (rot) + 1 (cp - cv correction for h = cp*T)
        # Actually h = (5/2 + 1)*R*T = 7/2*R*T for trans+rot (enthalpy, not energy)
        x = THETA_V_N2 / max(T, 1.0)
        if x > 500:
            h_vib = 0.0  # avoid overflow
        else:
            h_vib = R_N2 * THETA_V_N2 / (np.exp(x) - 1.0)
        return R_N2 * 3.5 * T + h_vib  # no formation enthalpy for N2 (reference)
    elif species == 'N':
        # Monatomic: h = (5/2)*R*T + h_formation
        return R_N * 2.5 * T + H_FORM_N
    else:
        raise ValueError(f"Unknown species: {species}")


def species_cp(species, T):
    """
    Species cp in J/(kg K).
    RRHO model: trans + rot + vib (harmonic).
    """
    if species == 'N2':
        # cp = 7/2*R + R*(theta/T)^2 * exp(theta/T) / (exp(theta/T)-1)^2
        x = THETA_V_N2 / max(T, 1.0)
        if x > 500:
            cp_vib = 0.0
        else:
            ex = np.exp(x)
            cp_vib = R_N2 * x**2 * ex / (ex - 1.0)**2
        return R_N2 * 3.5 + cp_vib
    elif species == 'N':
        return R_N * 2.5  # monatomic, constant
    else:
        raise ValueError(f"Unknown species: {species}")


def mixture_h(T, Y_N, Y_N2):
    """Mixture enthalpy in J/kg."""
    return Y_N * species_h('N', T) + Y_N2 * species_h('N2', T)


def mixture_cp(T, Y_N, Y_N2):
    """Mixture cp in J/(kg K)."""
    return Y_N * species_cp('N', T) + Y_N2 * species_cp('N2', T)


# ============================================================
# CFmesh Parser
# ============================================================

def parse_cfmesh(filepath):
    """Parse a COOLFluiD CFmesh file. Returns structured data."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = {
        'lines': lines,
        'nb_dim': 2, 'nb_eq': 5,
        'nb_nodes': 0, 'nb_states': 0, 'nb_elem': 0,
    }

    for line in lines:
        s = line.strip()
        parts = s.split()
        if len(parts) < 2:
            continue
        key = parts[0]
        if key == '!NB_DIM':
            data['nb_dim'] = int(parts[1])
        elif key == '!NB_EQ':
            data['nb_eq'] = int(parts[1])
        elif key == '!NB_NODES':
            data['nb_nodes'] = int(parts[1])
        elif key == '!NB_STATES':
            data['nb_states'] = int(parts[1])
        elif key == '!NB_ELEM':
            data['nb_elem'] = int(parts[1])

    # Parse nodes
    nd_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('!LIST_NODE'):
            nd_start = i + 1; break
    nodes = np.zeros((data['nb_nodes'], data['nb_dim']))
    for n in range(data['nb_nodes']):
        parts = lines[nd_start + n].strip().split()
        for d in range(data['nb_dim']):
            nodes[n, d] = float(parts[d])
    data['nodes'] = nodes
    data['nodes_line_start'] = nd_start

    # Parse elements
    el_start = None
    for i, line in enumerate(lines):
        if line.strip() == '!LIST_ELEM':
            el_start = i + 1; break
    elems = []
    for e in range(data['nb_elem']):
        elems.append([int(x) for x in lines[el_start + e].strip().split()])
    data['elems'] = elems
    data['elems_line_start'] = el_start

    # Parse states
    st_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('!LIST_STATE'):
            st_start = i + 1; break
    states = np.zeros((data['nb_states'], data['nb_eq']))
    for s in range(data['nb_states']):
        parts = lines[st_start + s].strip().split()
        for eq in range(data['nb_eq']):
            states[s, eq] = float(parts[eq])
    data['states'] = states
    data['states_line_start'] = st_start

    # Parse TRS
    data['trs'] = {}
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith('!TRS_NAME'):
            trs_name = lines[i].strip().split()[-1]
            trs_data = {'name': trs_name, 'line': i}
            for j in range(i, min(i + 10, len(lines))):
                if '!NB_GEOM_ENTS' in lines[j].strip():
                    trs_data['nb_faces'] = int(lines[j].strip().split()[-1])
                    break
            for j in range(i, min(i + 10, len(lines))):
                if lines[j].strip() == '!LIST_GEOM_ENT':
                    trs_data['faces_line_start'] = j + 1; break
            if 'faces_line_start' in trs_data and 'nb_faces' in trs_data:
                faces = []
                for f in range(trs_data['nb_faces']):
                    faces.append([int(x) for x in
                                  lines[trs_data['faces_line_start'] + f].strip().split()])
                trs_data['faces'] = faces
            data['trs'][trs_name] = trs_data
        i += 1
    return data


# ============================================================
# Domain Identification
# ============================================================

def identify_upstream_domain(data):
    """Identify upstream domain states via BFS from ShockUp nodes."""
    elems = data['elems']
    su_trs = data['trs'].get('ShockUp')
    if su_trs is None:
        print("ERROR: ShockUp TRS not found!"); sys.exit(1)

    su_nodes = set()
    for face in su_trs['faces']:
        for nid in face[2:]:
            su_nodes.add(nid)

    upstream_nodes = set(su_nodes)
    upstream_elems = set()
    upstream_states = set()

    changed = True
    while changed:
        changed = False
        for elem_id, elem in enumerate(elems):
            if elem_id in upstream_elems:
                continue
            if set(elem[:4]) & upstream_nodes:
                upstream_elems.add(elem_id)
                upstream_nodes |= set(elem[:4])
                for sid in elem[4:]:
                    upstream_states.add(sid)
                changed = True

    return upstream_states, upstream_elems


# ============================================================
# Step 1: Fix Upstream Initialization
# ============================================================

def fix_upstream_states(data, freestream, output_path):
    """Set upstream domain states to freestream, write cleaned CFmesh."""
    upstream_states, upstream_elems = identify_upstream_domain(data)
    states = data['states']

    n_upstream = len(upstream_states)
    up_T = states[sorted(upstream_states), 4]
    contaminated = (up_T > 2500).sum()
    print(f"Upstream domain: {len(upstream_elems)} elements, {n_upstream} states")
    print(f"Contaminated (T > 2500 K): {contaminated}/{n_upstream}")

    for sid in upstream_states:
        for eq in range(data['nb_eq']):
            states[sid, eq] = freestream[eq]

    lines = list(data['lines'])
    st_start = data['states_line_start']
    for s in range(data['nb_states']):
        vals = ' '.join(f'{states[s, eq]:.16e}' for eq in range(data['nb_eq']))
        lines[st_start + s] = vals + '\n'

    with open(output_path, 'w') as f:
        f.writelines(lines)

    print(f"Wrote: {output_path}")
    print(f"  {n_upstream} upstream → freestream, {data['nb_states'] - n_upstream} downstream kept")
    return upstream_states


# ============================================================
# Step 2: Real-Gas R-H Solver
# ============================================================

def compute_face_normals(data):
    """Compute outward face normals for ShockDown faces."""
    sd_trs = data['trs'].get('ShockDown')
    if sd_trs is None:
        print("ERROR: ShockDown TRS not found!"); sys.exit(1)

    nodes = data['nodes']
    elems = data['elems']

    face_data = []
    for face_idx, face in enumerate(sd_trs['faces']):
        nA, nB = face[2], face[3]
        pA, pB = nodes[nA], nodes[nB]
        dx, dy = pB[0] - pA[0], pB[1] - pA[1]
        L = np.sqrt(dx**2 + dy**2)
        nx_c, ny_c = -dy / L, dx / L

        for elem_id, elem in enumerate(elems):
            if nA in elem[:4] and nB in elem[:4]:
                centroid = nodes[elem[:4]].mean(axis=0)
                fc = (pA + pB) / 2.0
                if np.dot([nx_c, ny_c], centroid - fc) > 0:
                    nx, ny = -nx_c, -ny_c
                else:
                    nx, ny = nx_c, ny_c
                face_data.append({
                    'face_idx': face_idx, 'center': fc,
                    'normal': np.array([nx, ny]), 'length': L,
                })
                break

    face_data.sort(key=lambda f: f['center'][1])
    return face_data


def solve_rh_real_gas(rho_fs, u_n_fs, p_fs, T_fs, Y_N, Y_N2, R_mix,
                      max_iter=100, tol=1e-10):
    """
    Solve R-H jump conditions for a real gas (Vs=0).

    Given upstream normal state (rho1, un1, p1, T1, Ys),
    find downstream state (rho2, un2, p2, T2).

    Uses Newton iteration on rho2.
    Enthalpy from NASA-7 polynomials (includes formation enthalpy).
    """
    # Mass flux (constant across shock)
    m = rho_fs * u_n_fs  # kg/(m^2 s), negative (inflow)

    # Upstream enthalpy
    h1 = mixture_h(T_fs, Y_N, Y_N2)

    # Initial guess: use frozen gamma estimate
    cp_fs = mixture_cp(T_fs, Y_N, Y_N2)
    gamma_fs = cp_fs / (cp_fs - R_mix)
    M_n = abs(u_n_fs) / np.sqrt(gamma_fs * R_mix * T_fs)
    rho_ratio_guess = (gamma_fs + 1) * M_n**2 / ((gamma_fs - 1) * M_n**2 + 2)
    rho2 = rho_fs * rho_ratio_guess

    for it in range(max_iter):
        # From mass conservation
        u_n2 = m / rho2

        # From momentum conservation
        p2 = p_fs + m * (u_n_fs - u_n2)  # = p1 + rho1*un1^2 - rho2*un2^2

        # From energy conservation
        h2_target = h1 + 0.5 * (u_n_fs**2 - u_n2**2)

        # Temperature from EOS
        T2 = p2 / (rho2 * R_mix)

        if T2 < 100.0:
            T2 = 100.0
            p2 = rho2 * R_mix * T2

        # Enthalpy at T2
        h2_actual = mixture_h(T2, Y_N, Y_N2)

        # Residual
        residual = h2_actual - h2_target

        if abs(residual) < tol * abs(h2_target):
            break

        # Derivative dResidual/drho2 (finite difference)
        drho = max(abs(rho2) * 1e-6, 1e-12)
        rho2p = rho2 + drho
        u_n2p = m / rho2p
        p2p = p_fs + m * (u_n_fs - u_n2p)
        h2_target_p = h1 + 0.5 * (u_n_fs**2 - u_n2p**2)
        T2p = p2p / (rho2p * R_mix)
        T2p = max(T2p, 100.0)
        h2_actual_p = mixture_h(T2p, Y_N, Y_N2)
        residual_p = h2_actual_p - h2_target_p

        dres_drho = (residual_p - residual) / drho
        if abs(dres_drho) < 1e-30:
            break

        # Newton update with damping
        delta = -residual / dres_drho
        rho2 += 0.8 * delta
        rho2 = max(rho2, rho_fs * 1.01)  # must be denser than upstream

    return rho2, u_n2, p2, T2, it


def compute_rh_postshock(face_data, freestream, R_mix):
    """Compute real-gas R-H post-shock state at each ShockDown face."""
    rho_N_fs, rho_N2_fs, u_fs, v_fs, T_fs = freestream
    rho_fs = rho_N_fs + rho_N2_fs
    Y_N = rho_N_fs / rho_fs
    Y_N2 = rho_N2_fs / rho_fs
    p_fs = rho_fs * R_mix * T_fs

    # Verify freestream enthalpy
    h_fs = mixture_h(T_fs, Y_N, Y_N2)
    V_fs = np.sqrt(u_fs**2 + v_fs**2)
    h0_fs = h_fs + 0.5 * V_fs**2
    cp_fs = mixture_cp(T_fs, Y_N, Y_N2)
    gamma_fs = cp_fs / (cp_fs - R_mix)
    a_fs = np.sqrt(gamma_fs * R_mix * T_fs)

    print(f"Freestream: rho={rho_fs:.5f}, u={u_fs:.0f}, T={T_fs:.0f}")
    print(f"  p={p_fs:.1f} Pa, h={h_fs:.0f} J/kg, h0={h0_fs:.0f} J/kg")
    print(f"  cp={cp_fs:.1f}, gamma={gamma_fs:.4f}, a={a_fs:.0f}, M={V_fs/a_fs:.2f}")
    print(f"  Y_N={Y_N:.4f}, Y_N2={Y_N2:.4f}, R_mix={R_mix:.2f}")

    results = []
    for fd in face_data:
        nx, ny = fd['normal']

        # Decompose freestream velocity
        u_n_fs_loc = u_fs * nx + v_fs * ny  # negative (inflow)
        u_t_fs_loc = -u_fs * ny + v_fs * nx  # tangential (preserved)

        M_n = abs(u_n_fs_loc) / a_fs

        # Solve real-gas R-H
        rho2, u_n2, p2, T2, n_iter = solve_rh_real_gas(
            rho_fs, u_n_fs_loc, p_fs, T_fs, Y_N, Y_N2, R_mix)

        # Post-shock sound speed and Mach
        cp2 = mixture_cp(T2, Y_N, Y_N2)
        gamma2 = cp2 / (cp2 - R_mix)
        a2 = np.sqrt(gamma2 * R_mix * T2)
        M_n2 = abs(u_n2) / a2

        # Convert back to global frame
        u_post = u_n2 * nx - u_t_fs_loc * ny
        v_post = u_n2 * ny + u_t_fs_loc * nx

        V_post = np.sqrt(u_post**2 + v_post**2)
        M_total = V_post / a2

        # Species densities (frozen composition)
        rho_N_post = Y_N * rho2
        rho_N2_post = Y_N2 * rho2

        # Verify energy conservation
        h2 = mixture_h(T2, Y_N, Y_N2)
        h0_post = h2 + 0.5 * (u_n2**2 + u_t_fs_loc**2)
        h0_err = abs(h0_post - h0_fs) / h0_fs

        results.append({
            'y': fd['center'][1],
            'rho_N': rho_N_post, 'rho_N2': rho_N2_post,
            'u': u_post, 'v': v_post, 'T': T2,
            'M_n': M_n, 'M_n_post': M_n2, 'M_total_post': M_total,
            'rho_ratio': rho2 / rho_fs,
            'u_n_post': u_n2, 'p_post': p2,
            'gamma_post': gamma2, 'n_iter': n_iter,
            'h0_err': h0_err,
            'face_idx': fd['face_idx'],
        })

    return results


def write_rh_file(results, output_path):
    """Write R-H state in BCDirichletFromFile format."""
    with open(output_path, 'w') as f:
        f.write("y rho_N rho_N2 u v T\n")
        f.write(f"{len(results)}\n")
        for r in results:
            f.write(f"{r['y']:.10e} {r['rho_N']:.10e} {r['rho_N2']:.10e} "
                    f"{r['u']:.10e} {r['v']:.10e} {r['T']:.10e}\n")
    print(f"Wrote: {output_path} ({len(results)} points)")


def print_rh_summary(results):
    """Print summary table."""
    print()
    print("=" * 130)
    print("R-H POST-SHOCK STATE (REAL GAS, NASA-7 polynomials, frozen composition, Vs=0)")
    print("=" * 130)
    print(f"{'idx':>3} {'y':>8} {'M_n':>6} {'rho_r':>6} {'rho_N':>10} {'rho_N2':>10} "
          f"{'u':>9} {'v':>9} {'T':>8} {'p':>8} {'u_n':>8} {'M_n2':>6} {'M_tot':>6} "
          f"{'gam2':>5} {'it':>3} {'h0err':>8}")

    for r in results:
        print(f"{r['face_idx']:3d} {r['y']:8.4f} {r['M_n']:6.2f} {r['rho_ratio']:6.2f} "
              f"{r['rho_N']:10.6f} {r['rho_N2']:10.6f} {r['u']:9.1f} {r['v']:9.1f} "
              f"{r['T']:8.0f} {r['p_post']:8.0f} {r['u_n_post']:8.1f} "
              f"{r['M_n_post']:6.3f} {r['M_total_post']:6.3f} "
              f"{r['gamma_post']:5.3f} {r['n_iter']:3d} {r['h0_err']:8.2e}")

    # Stagnation face summary
    stag = [r for r in results if abs(r['y']) < 0.001]
    if stag:
        s = stag[0]
        print()
        print(f"Stagnation (face {s['face_idx']}): "
              f"T_post={s['T']:.0f} K, p_post={s['p_post']:.0f} Pa, "
              f"rho_post={s['rho_N']+s['rho_N2']:.5f} kg/m3, "
              f"u_n_post={s['u_n_post']:.1f} m/s, M_n_post={s['M_n_post']:.3f}, "
              f"gamma_post={s['gamma_post']:.3f}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fix upstream + compute real-gas R-H post-shock state")
    parser.add_argument('--cfmesh', default='shockfitting_v2_P1.CFmesh')
    parser.add_argument('--output-cfmesh', default='shockfitting_v2_P1_v2.CFmesh')
    parser.add_argument('--output-rh', default='RH_postshock_v2.dat')
    args = parser.parse_args()

    freestream = [0.0001952, 0.004956, 5590.0, 0.0, 1833.0]
    R_mix = 308.05

    print(f"Reading: {args.cfmesh}")
    data = parse_cfmesh(args.cfmesh)
    print(f"  {data['nb_elem']} elements, {data['nb_states']} states")

    # Step 1
    print("\n" + "=" * 60)
    print("STEP 1: Fix upstream domain")
    print("=" * 60)
    fix_upstream_states(data, freestream, args.output_cfmesh)

    # Step 2
    print("\n" + "=" * 60)
    print("STEP 2: Compute REAL-GAS R-H post-shock state")
    print("=" * 60)
    face_data = compute_face_normals(data)
    results = compute_rh_postshock(face_data, freestream, R_mix)
    write_rh_file(results, args.output_rh)
    print_rh_summary(results)

    # Dense file (midpoints for smoother interpolation)
    if len(results) > 1:
        dense = []
        for i in range(len(results)):
            dense.append(results[i])
            if i < len(results) - 1:
                mid = {}
                for key in ['y', 'rho_N', 'rho_N2', 'u', 'v', 'T']:
                    mid[key] = 0.5 * (results[i][key] + results[i + 1][key])
                dense.append(mid)
        dp = args.output_rh.replace('.dat', '_dense.dat')
        write_rh_file(dense, dp)

    print("\nDone.")


if __name__ == '__main__':
    main()
