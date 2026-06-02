#!/usr/bin/env python3
"""
SHOCK FITTING LOOP — Step 1: Compute New Shock Contour
=======================================================

Reads a converged CFmesh solution, computes the R-H shock velocity Vs at
each ShockDown face, and moves the shock contour by Vs * dt.

Outputs:
  - new_shock_contour.dat       — (x, y) points for the moved shock
  - shock_contour_for_mesh.dat  — format compatible with hornung_shockfitted_v2_mesh.py
  - shock_velocity_report.txt   — detailed Vs diagnostic
  - shock_comparison.png        — old vs new shock contour overlay plot

Usage:
  python3 step1_compute_new_shock_contour.py --cfmesh /path/to/solution.CFmesh [--relax 0.5] [--target-dx 0.0001] [--symmetrize]

After running:
  1. Check shock_comparison.png — verify displacement looks reasonable
  2. Copy shock_contour_for_mesh.dat → Hornung_cylinedr_mesh/shockfitting/shock_contour.dat
  3. cd Hornung_cylinedr_mesh/shockfitting/
  4. python3 hornung_shockfitted_v2_mesh.py  → shockfitting_v2_base.msh (SINGLE domain)
  5. python3 split_at_b23.py                 → shockfitting_v2.msh (TWO domains: ShockDown/ShockUp)
  6. python3 interpolate_p0.py               → shockfitting_interp_v2.CFmesh
  7. cd back to Hornung_N2_SF/
  8. ./coolfluid-solver --scase MeshUpgrade_P1.CFcase  → shockfitting_v2_P1.CFmesh
  9. Run sf_init.py (first time) or sf_adjust.py (with --old-solution) on the new P1 mesh
"""

import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
from fix_upstream_and_compute_rh_v2 import parse_cfmesh, compute_face_normals


def main():
    parser = argparse.ArgumentParser(description="Step 1: Compute new shock contour from Vs")
    parser.add_argument('--cfmesh', required=True, help='Converged CFmesh solution file')
    parser.add_argument('--dt', type=float, default=0.0,
                        help='Time step for shock displacement (s). If 0, auto-computed.')
    parser.add_argument('--relax', type=float, default=0.5,
                        help='Relaxation factor for shock displacement (0-1)')
    parser.add_argument('--target-dx', type=float, default=0.0001,
                        help='Target max displacement (m) for auto dt computation')
    parser.add_argument('--outdir', default='.', help='Output directory')
    parser.add_argument('--symmetrize', action='store_true',
                        help='Force top/bottom symmetry on the moved contour. '
                             'Off by default — the raw asymmetric Vs correction is preserved.')
    args = parser.parse_args()

    # Fix #3: create output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)

    # Freestream
    rho_N_fs, rho_N2_fs, u_fs, v_fs, T_fs = 0.0001952, 0.004956, 5590.0, 0.0, 1833.0
    rho_fs = rho_N_fs + rho_N2_fs

    # Parse mesh
    print(f"Reading: {args.cfmesh}")
    data = parse_cfmesh(args.cfmesh)
    nodes = data['nodes']
    arr = data['states']
    elems = data['elems']
    print(f"  {data['nb_elem']} elements, {data['nb_states']} states")

    # Get ShockDown face normals and geometry
    face_data = compute_face_normals(data)

    # Map faces to elements
    sd_trs = data['trs']['ShockDown']
    sd_faces_raw = [(f[2], f[3]) for f in sd_trs['faces']]
    sd_elem_map = {}
    for elem_id, elem in enumerate(elems):
        enodes = set(elem[:4])
        for fi, (nA, nB) in enumerate(sd_faces_raw):
            if nA in enodes and nB in enodes:
                sd_elem_map[fi] = (elem_id, elem[4:])
                break

    m_nbSpecies = data['nb_eq'] - 3

    # Cell-average state evaluation for shock velocity diagnostic.
    #
    # Previously this used exact FR polynomial evaluation at face flux points
    # (extrapolating to eta=1.0 on the reference element boundary). That
    # approach amplifies tiny numerical asymmetry from MPI decomposition /
    # iterative solver into large Vs asymmetry (e.g. face 5: +155 vs face 19:
    # -42 on a symmetric problem). The cell-average gives a robust, symmetric
    # result that represents the bulk mass balance through the shock cell.

    # Compute Vs at each face
    results = []
    for fd in face_data:
        fi = fd['face_idx']
        if fi not in sd_elem_map:
            continue
        nx, ny = fd['normal']
        elem_id, state_ids = sd_elem_map[fi]

        # Cell-average state (robust for shock velocity diagnostic)
        elem = elems[elem_id]
        estates = elem[4:]
        s_avg = arr[estates].mean(axis=0)

        rho_int = sum(s_avg[i] for i in range(m_nbSpecies))
        u_int, v_int = s_avg[m_nbSpecies], s_avg[m_nbSpecies + 1]
        u_n_int = u_int * nx + v_int * ny
        u_n_fs = u_fs * nx + v_fs * ny
        mflux_fs = rho_fs * u_n_fs
        mflux_int = rho_int * u_n_int
        drho = rho_fs - rho_int
        Vs = (mflux_fs - mflux_int) / drho if abs(drho) > 1e-30 else 0.0

        # Current shock node positions (face midpoint)
        nA, nB = sd_faces_raw[fi]
        x_mid = 0.5 * (nodes[nA][0] + nodes[nB][0])
        y_mid = 0.5 * (nodes[nA][1] + nodes[nB][1])

        results.append({
            'fi': fi, 'x': x_mid, 'y': y_mid,
            'nx': nx, 'ny': ny, 'Vs': Vs,
            'mflux_fs': mflux_fs, 'mflux_int': mflux_int,
            'rho_int': rho_int, 'u_n_int': u_n_int,
            'T_int': s_avg[m_nbSpecies + 2],
        })

    # Sort by y
    results.sort(key=lambda r: r['y'])

    # Auto dt: scale so max displacement = target_dx
    Vs_max = max(abs(r['Vs']) for r in results)
    if args.dt <= 0:
        if Vs_max > 1e-10:
            dt = args.target_dx / Vs_max
        else:
            dt = 1e-5
        print(f"Auto dt = {dt:.6e} s (target_dx={args.target_dx}, max|Vs|={Vs_max:.1f})")
    else:
        dt = args.dt
        print(f"Using dt = {dt:.6e} s")

    relax = args.relax
    print(f"Relaxation = {relax}")

    # Compute displacements
    new_contour = []
    for r in results:
        dx = relax * r['Vs'] * dt * r['nx']
        dy = relax * r['Vs'] * dt * r['ny']
        x_new = r['x'] + dx
        y_new = r['y'] + dy
        new_contour.append({'x': x_new, 'y': y_new,
                            'x_old': r['x'], 'y_old': r['y'],
                            'Vs': r['Vs'], 'dx': dx, 'dy': dy})

    # Optionally enforce symmetry (off by default)
    if args.symmetrize:
        print("\n  Symmetrizing contour (--symmetrize flag)")
        n = len(new_contour)
        mid = n // 2
        for k in range(1, mid + 1):
            i_bot = mid - k
            i_top = mid + k
            if i_bot >= 0 and i_top < n:
                x_avg = 0.5 * (new_contour[i_bot]['x'] + new_contour[i_top]['x'])
                new_contour[i_bot]['x'] = x_avg
                new_contour[i_top]['x'] = x_avg
                y_avg = 0.5 * (abs(new_contour[i_bot]['y']) + abs(new_contour[i_top]['y']))
                new_contour[i_bot]['y'] = -y_avg
                new_contour[i_top]['y'] = y_avg
        for c in new_contour:
            c['dx'] = c['x'] - c['x_old']
            c['dy'] = c['y'] - c['y_old']

    # Print table (after symmetrization if active, so values match written output)
    print()
    print(f"{'fi':>3} {'y':>8} {'Vs':>8} {'dx':>10} {'dy':>10} {'|disp|':>10}  {'x_old':>10} {'x_new':>10}")
    for i, r in enumerate(results):
        c = new_contour[i]
        disp = np.sqrt(c['dx']**2 + c['dy']**2)
        print(f"{r['fi']:3d} {c['y']:8.4f} {r['Vs']:8.1f} {c['dx']:10.6f} {c['dy']:10.6f} {disp:10.6f}  {c['x_old']:10.6f} {c['x']:10.6f}")

    # Write new shock contour
    contour_path = os.path.join(args.outdir, 'new_shock_contour.dat')
    with open(contour_path, 'w') as f:
        f.write(f"# New shock contour after Vs displacement (dt={dt:.6e}, relax={relax})\n")
        f.write(f"# max|Vs|={Vs_max:.2f} m/s, max|displacement|={max(np.sqrt(c['dx']**2+c['dy']**2) for c in new_contour):.6f} m\n")
        f.write(f"# x  y  x_old  y_old  Vs  dx  dy\n")
        f.write(f"{len(new_contour)}\n")
        for c in new_contour:
            f.write(f"{c['x']:.12e} {c['y']:.12e} {c.get('x_old',0):.12e} {c.get('y_old',0):.12e} "
                    f"{c.get('Vs',0):.6e} {c.get('dx',0):.6e} {c.get('dy',0):.6e}\n")
    print(f"\nWrote: {contour_path}")

    # Write Gmsh snippet for B23 points
    geo_path = os.path.join(args.outdir, 'new_B23_points.geo')
    with open(geo_path, 'w') as f:
        f.write("// --- B23 = SHOCK CONTOUR (moved by Vs, bottom to top) ---\n")
        f.write(f"// max|Vs|={Vs_max:.2f} m/s, dt={dt:.6e}, relax={relax}\n")
        f.write(f"N_pts = {len(new_contour)};\n")
        for idx, c in enumerate(new_contour):
            f.write(f"Point({3000 + idx}) = {{{c['x']:.12e}, {c['y']:.12e}, 0, lc}};\n")
    print(f"Wrote: {geo_path}")

    # Write report
    report_path = os.path.join(args.outdir, 'shock_velocity_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Shock Velocity Report\n")
        f.write(f"Input: {args.cfmesh}\n")
        f.write(f"dt={dt:.6e}, relax={relax}\n")
        f.write(f"max|Vs|={Vs_max:.2f} m/s\n")
        f.write(f"mean|Vs|={np.mean([abs(r['Vs']) for r in results]):.2f} m/s\n\n")
        f.write(f"{'fi':>3} {'y':>8} {'Vs':>8} {'mflux_fs':>10} {'mflux_int':>10} {'imbal%':>8} "
                f"{'rho_int':>8} {'u_n_int':>8} {'T_int':>7}\n")
        for r in results:
            imbal = 100 * (r['mflux_fs'] - r['mflux_int']) / abs(r['mflux_fs']) if abs(r['mflux_fs']) > 1e-30 else 0
            f.write(f"{r['fi']:3d} {r['y']:8.4f} {r['Vs']:8.1f} {r['mflux_fs']:10.3f} {r['mflux_int']:10.3f} "
                    f"{imbal:8.1f}% {r['rho_int']:8.5f} {r['u_n_int']:8.1f} {r['T_int']:7.0f}\n")
    print(f"Wrote: {report_path}")

    # Write contour in shock_contour.dat format (for hornung_shockfitted_v2_mesh.py)
    # Format: x  y  theta  r (sorted top to bottom, ie theta from +pi/2 to -pi/2)
    mesh_contour_path = os.path.join(args.outdir, 'shock_contour_for_mesh.dat')
    with open(mesh_contour_path, 'w') as f:
        f.write(f"# New shock contour (moved by Vs, for hornung_shockfitted_v2_mesh.py)\n")
        f.write(f"# Source: {args.cfmesh}\n")
        f.write(f"# dt={dt:.6e}, relax={relax}, max|Vs|={Vs_max:.2f} m/s\n")
        f.write(f"# x[m]  y[m]  theta[rad]  r[m]\n")
        # Reverse order (top to bottom) for compatibility
        for c in reversed(new_contour):
            r = np.sqrt(c['x']**2 + c['y']**2)
            theta = np.arctan2(c['y'], c['x'])
            f.write(f"{c['x']:.12e} {c['y']:.12e} {theta:.12e} {r:.12e}\n")
    print(f"Wrote: {mesh_contour_path}")

    # --- Comparison plot: old vs new shock contour ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left panel: full view
        ax = axes[0]
        old_x = [c.get('x_old', c['x']) for c in new_contour]
        old_y = [c.get('y_old', c['y']) for c in new_contour]
        new_x = [c['x'] for c in new_contour]
        new_y = [c['y'] for c in new_contour]

        ax.plot(old_x, old_y, 'b-o', lw=2, ms=4, label='Old shock (v2)')
        ax.plot(new_x, new_y, 'r-s', lw=2, ms=4, label='New shock (moved)')

        # Draw displacement arrows
        for c in new_contour:
            x_old = c.get('x_old', c['x'])
            y_old = c.get('y_old', c['y'])
            if abs(c.get('dx', 0)) > 1e-8 or abs(c.get('dy', 0)) > 1e-8:
                ax.annotate('', xy=(c['x'], c['y']), xytext=(x_old, y_old),
                            arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

        # Cylinder
        theta_cyl = np.linspace(0, 2*np.pi, 200)
        R_cyl = 0.01275
        ax.fill(R_cyl*np.cos(theta_cyl), R_cyl*np.sin(theta_cyl), color='#555', zorder=3)

        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(f'Shock Contour Comparison\nmax|Vs|={Vs_max:.1f} m/s')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

        # Right panel: Vs and displacement along shock
        ax2 = axes[1]
        ys = [c.get('y_old', c['y']) for c in new_contour]
        vs_vals = [c.get('Vs', 0) for c in new_contour]
        disp_vals = [np.sqrt(c.get('dx',0)**2 + c.get('dy',0)**2)*1000 for c in new_contour]

        color1 = 'tab:blue'
        ax2.plot(ys, vs_vals, 'o-', color=color1, lw=2, ms=5, label='Vs (m/s)')
        ax2.set_xlabel('y [m]')
        ax2.set_ylabel('Shock Velocity Vs [m/s]', color=color1)
        ax2.tick_params(axis='y', labelcolor=color1)
        ax2.axhline(y=0, color='k', ls='--', lw=0.5)

        ax3 = ax2.twinx()
        color2 = 'tab:red'
        ax3.plot(ys, disp_vals, 's-', color=color2, lw=2, ms=5, label='|disp| (mm)')
        ax3.set_ylabel('Displacement [mm]', color=color2)
        ax3.tick_params(axis='y', labelcolor=color2)

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax2.set_title('Shock Velocity & Displacement')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(args.outdir, 'shock_comparison.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Wrote: {plot_path}")
    except ImportError:
        print("WARNING: matplotlib not available, skipping comparison plot")

    print(f"\n=== SUMMARY ===")
    print(f"  max|Vs| = {Vs_max:.1f} m/s")
    print(f"  max displacement = {max(np.sqrt(c['dx']**2+c['dy']**2) for c in new_contour)*1000:.4f} mm")
    meshdir = "/mnt/c/Research/codes/Solvers/Hornung_cylinedr_mesh/shockfitting"
    print(f"\nNext steps:")
    print(f"  1. Check shock_comparison.png — verify the displacement looks reasonable")
    print(f"  2. Copy contour to mesh generation directory:")
    print(f"       cp {mesh_contour_path} {meshdir}/shock_contour.dat")
    print(f"  3. Regenerate mesh (TWO steps!):")
    print(f"       cd {meshdir}")
    print(f"       python3 hornung_shockfitted_v2_mesh.py   # → shockfitting_v2_base.msh")
    print(f"       python3 split_at_b23.py                   # → shockfitting_v2.msh (ShockDown/ShockUp)")
    print(f"  4. Interpolate P0 onto new mesh:")
    print(f"       python3 interpolate_p0.py                 # reads shockfitting_v2.msh")
    print(f"  5. Back in the SF test case directory, upgrade P0 → P1:")
    print(f"       ./coolfluid-solver --scase MeshUpgrade_P1.CFcase")
    print(f"  6. Prepare mesh:")
    print(f"       LD_LIBRARY_PATH=...plato.../lib:$LD_LIBRARY_PATH \\")
    print(f"       python3 shock_fitting_loop/sf_init.py \\")
    print(f"         --p1-mesh shockfitting_v2_P1.CFmesh --output-dir {args.outdir}")
    print(f"     Or with solution carryover (subsequent iterations):")
    print(f"       python3 shock_fitting_loop/sf_adjust.py \\")
    print(f"         --p1-mesh shockfitting_v2_P1.CFmesh \\")
    print(f"         --old-solution RESULTS_xxx/P1_CNEQ_SF-iter_XXXX.CFmesh \\")
    print(f"         --output-dir {args.outdir}")
    print(f"  7. Update CFcase, run simulation")


if __name__ == '__main__':
    main()
