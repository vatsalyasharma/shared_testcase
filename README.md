# shared_testcase — verified COOLFluiD FR test cases

A small, clean collection of Flux Reconstruction (FR) test cases that are confirmed to run on the merged build. 

Every case is two steps: (1) a mesh upgrade that builds the mesh for that order, then (2) the solve. 

All NEQ (CNEQ and TCNEQ) cases use the VS (variable-set) solver path and the RANS uses the physicality-VS limiter.

## How to run any case (general recipe)

Use the merged-build solver with absolute paths:

```bash
cd <case folder>
```
# 1) build the mesh for this order
```bash
mpirun -n 1 $SOLVER --scase ./<MeshUpgrade...CFcase> 
```
# 2) solve
```bash
mpirun -n 1 $SOLVER --scase ./<solve...CFcase>      
```
### Example A — TCNEQ P0 (full chain from the base mesh)
```bash
cd TCNEQ/Hornung_cylinder/P0
mpirun -n 1 $SOLVER --scase ./MeshUpgrade_P0_Q2.CFcase --ldir $DSO   # sample_4block_v17.CFmesh -> sample_4block_v17_P0_Q2.CFmesh
mpirun -n 1 $SOLVER --scase ./P0_TCNEQ_Q2.CFcase       --ldir $DSO   # solves on that mesh
```

### Example B — TCNEQ P1 (starts from a saved P0 solution)
```bash
cd TCNEQ/Hornung_cylinder/EF/P1
mpirun -n 1 $SOLVER --scase ./MeshUpgrade_P0_to_P1.CFcase --ldir $DSO  # sample_v17_P0_TCNEQ_Q2_10iter.CFmesh -> P1 mesh
mpirun -n 1 $SOLVER --scase ./P1_TCNEQ_Q2_EF.CFcase       --ldir $DSO  # solves the P1 case
```
CNEQ and RANS work the same way; just use the file names in each folder. The `.inter` file (when present) sets CFL etc. and can be edited while running. You may create your own inter files.

## Meshes kept for higher-order runs

- P0 folders in all cases have the base mesh (`sample_4block_v17.CFmesh` for the cylinder; the Gmsh `.msh` for the flat plate).
- Higher-order folders (P1/P3/P7) have the saved P0 solution they start from (e.g. `sample_v17_P0_TCNEQ_Q2_10iter.CFmesh`, `flatPlateGReKLogOCoarse_physVS_P0_iter200.CFmesh`) along with the the mesh-upgrade
  case. Always first run the upgrade to rebuild the mesh for P1/P3/P7 case in one command.
- Workflow: P0 -> upgrade -> P1 -> solve .

> Note: the P0 solutions in this tests are restarts of short runs (10-200 iters), good enough to start the higher-order solve. For your run, regenerate them from a fully converged P0.

## Status 

- Verified at 5 iterations (mesh upgrade + solve, clean exit, no errors): all CNEQ and TCNEQ cases (P0, EF, LLAV, Shock_fitting) and RANS p0/p1/p3/p7.

- Shock fitting treats the shock as a moving boundary. The shock-aligned mesh is shipped (`shockfitting_interp_v2.CFmesh` / `shockfitting_v2.msh`), so the solve runs directly. To *regenerate* the mesh and close the iterative loop, the mesh-generator scripts are in `/mnt/c/codes/Hornung_cylinedr_mesh/shockfitting/` (pipeline: `extract_shock.py` → `hornung_shockfitted_v2_mesh.py` → `split_at_b23.py` → `interpolate_p0.py`, starting from the base `shock_contour.dat`). Workflow details in `Shock_fitting/shock_fitting_loop/README.md`. The TCNEQ shock R-H data is approximate.
- Empty `EF/P2`, `LLAV/P2` folders under CNEQ are placeholders for future orders.
- Always change the paths inside the cases before using them.

## Shock fitting — step-by-step

Shock fitting does **not** capture the shock; it puts the shock on a mesh boundary. The mesh is split at the shock into two sides — `ShockDown` (downstream) and `ShockUp` (upstream) — and the post-shock state from the Rankine–Hugoniot (R-H) relations is imposed on `ShockDown` (`DirichletFromFile`, read from `RH_postshock.dat`). Everything is **CNEQ/TCNEQ P1** on the shock-aligned mesh; the same two-step pattern as the other cases (upgrade → solve).

```bash
SOLVER=/mnt/c/codes/COOLFluiD/COOLFluiD-merged/optim/apps/Solver/coolfluid-solver
```

### Run CNEQ shock fitting
```bash
cd CNEQ/Hornung_cylinder/Shock_fitting
# 1) upgrade the P0 shock solution to P1
mpirun -n 1 $SOLVER --scase ./MeshUpgrade_P1.CFcase 
#    shockfitting_interp_v2.CFmesh  ->  shockfitting_v2_P1.CFmesh
# 2) solve P1 with the R-H boundary on ShockDown
mpirun -n 1 $SOLVER --scase ./P1_CNEQ_SF.CFcase  
#    reads shockfitting_v2_P1.CFmesh + RH_postshock.dat ; CFL etc. in P1_CNEQ_SF.inter
```

### Run TCNEQ shock fitting
```bash
cd TCNEQ/Hornung_cylinder/Shock_fitting
# 1) upgrade the P0 shock solution to P1
mpirun -n 1 $SOLVER --scase ./MeshUpgrade_P1.CFcase 
#    shockfitting_interp_TCNEQ.CFmesh  ->  shockfitting_TCNEQ_P1.CFmesh
# 2) solve P1 with the R-H boundary on ShockDown
mpirun -n 1 $SOLVER --scase ./P1_TCNEQ_SF.CFcase   
#    reads shockfitting_TCNEQ_P1.CFmesh + RH_postshock.dat ; CFL in P1_TCNEQ_SF.inter
```

That is all that is needed to run the shipped cases (verified for 5 iters).

### How these cases were built (provenance)

CNEQ is the original; TCNEQ was derived from it.

- **The shock-aligned mesh** comes from the generator folder
  `/mnt/c/codes/Hornung_cylinedr_mesh/shockfitting/`:
  `extract_shock.py` (reads a converged P0 solution, finds the bow shock by its pressure jump) → `shock_contour.dat` → `hornung_shockfitted_v2_mesh.py` (base mesh) → `split_at_b23.py` (duplicates nodes at the shock to make  `ShockDown`/`ShockUp`) → `shockfitting_v2.msh` → `interpolate_p0.py` (puts the P0 solution on it) → `shockfitting_interp_v2.CFmesh`. That last file is the shipped CNEQ starting point.
- **R-H data** (`RH_postshock.dat`) is produced by `tools/fix_upstream_and_compute_rh_v2.py` (post-shock state from the R-H jump relations along the shock line).
- **TCNEQ from CNEQ:** the CNEQ P0 shock mesh was extended from 5 to 6 equations  (added `Tv = T`) → `shockfitting_interp_TCNEQ.CFmesh`; a `Tv` column (frozen freestream `1833 K`) was added to the R-H file; and the case was   switched to the TCNEQ VS path (`TNEQSourceTermVS`, `Euler2DNEQConsToRhoivtTvInRhoivtTvVS`, 6-component state). This R-H data is therefore **approximate** (single-T `T` + frozen `Tv`); regenerate it with a 2-temperature R-H for production.
- The in-solver `ComputeShockVelocity` diagnostic is not in this build, so it was removed from the cases; compute the shock velocity with `shock_fitting_loop/step1_compute_new_shock_contour.py` instead.

### Moving the shock (the full iterative loop)

To drive the shock to its correct position (`Vs → 0`), repeat:

solve → `shock_fitting_loop/step1_compute_new_shock_contour.py` (new contour from the converged solution) → regenerate the mesh with the generator folder above → `interpolate_p0.py` → `MeshUpgrade_P1` → `sf_init.py` `sf_adjust.py` → solve again. Stop when `max|Vs|` is small (~10 m/s). 
