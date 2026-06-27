# `SolverBoundaryImpulse` — boundary-impulse coupling of reduced and maximal coordinates

> **Experimental.** This is a proof-of-concept. API, math, defaults, and file
> layout may change without notice. It lives under
> `newton/_src/solvers/experimental/` precisely so it does *not* yet carry the
> stability guarantees of the public solvers.

## 1. Why this solver exists

Newton ships excellent single-representation solvers:

* **Reduced / generalized coordinate** solvers (`SolverFeatherstone`,
  `SolverMuJoCo`) integrate articulated trees in joint space `q`, `v`. They are
  fast and drift-free for open kinematic chains (serial robot arms), because the
  tree topology is baked into the coordinates: there are exactly as many DOFs as
  the mechanism actually has, and joint constraints are satisfied *exactly* by
  construction.

* **Maximal / body coordinate** solvers (`SolverKamino`, `SolverXPBD`,
  `SolverSemiImplicit`) integrate every body's full 6-DOF pose and resolve joints
  as explicit constraints. They handle **closed kinematic loops** (four-bar
  linkages, parallelogram grippers, the Robotiq 2F-85's coupled fingers)
  naturally, because a loop is just one more constraint — whereas a reduced
  solver must either cut the loop and re-close it with a Lagrange constraint
  (losing its main advantage) or cannot represent it at all.

A realistic manipulation system is *both at once*: a serial arm (open chain, best
in reduced coordinates) carrying a closed-chain gripper (best in maximal
coordinates). `SolverBoundaryImpulse` lets each subsystem keep the solver that
suits it and couples them through a single shared rigid boundary, exchanging a
physically meaningful **6D boundary impulse** so that Newton's third law holds
across the seam.

## 2. What problem it solves

Given:

* a **reduced** subsystem with generalized state `(q, v)`, end-effector body
  `e`, integrated by some reduced solver `R`;
* a **maximal** subsystem with body state `(X, V)`, base body `b`, integrated by
  some maximal solver `M`;
* a rigid attachment that welds a frame on `e` to a frame on `b`.

we want to advance both subsystems one step so that, at the welded boundary, the
two subsystems share a velocity (and, with stabilization, a pose) and the
coupling force is *equal and opposite* on the two sides. The solver computes a
boundary impulse `λ ∈ ℝ⁶` and applies `+Jᵣᵀλ` to the reduced side and `−Jₘᵀλ`
to the maximal side.

## 3. Why full reduced-coordinate simulation is insufficient for closed-chain grippers

A reduced-coordinate solver parameterizes the system by a spanning tree of
joints. A closed loop (the defining feature of the 2F-85 gripper and of any
four-bar) has *more* joints than the tree can hold: the loop-closure joint is not
representable as an independent generalized coordinate. The usual reduced-solver
remedies are:

* **Cut the loop** and re-impose closure with an explicit bilateral constraint,
  solved at velocity/acceleration level each step. This drags the reduced solver
  back into the same constraint-stabilization regime as a maximal solver, eroding
  the very drift-free exactness that justified using reduced coordinates — and
  Newton's reduced solvers do not expose a general loop-closure constraint API.
* **Approximate the loop** with stiff springs (a "squishy" four-bar). This is what
  the existing Kamino four-bar example does for *visualization*, but it changes the
  mechanism's dynamics.

In short: serial arms are a perfect fit for reduced coordinates; closed-chain
grippers are not.

## 4. Why full maximal-coordinate simulation may be slower for serial arms

Running the *whole* system (arm + gripper) in maximal coordinates is correct but
pays for the arm's open chain twice over:

* Every arm link gets 6 DOFs plus an explicit joint constraint, where reduced
  coordinates need only 1 DOF per revolute joint and **zero** constraint work.
  A 7-DOF arm becomes ~42 body DOFs + ~36 constraint rows instead of 7 clean DOFs.
* Maximal solvers must iterate (PADMM/PBD/constraint sweeps) to drive joint
  violations toward zero each step; reduced solvers satisfy those joints exactly
  in one linear solve. For a stiff serial arm this means smaller timesteps or more
  iterations for equivalent accuracy.

So neither pure representation is ideal for the mixed system. The point of this
solver is to use each representation where it is strong.

## 5. What is coupled

A single **6D rigid boundary constraint** between the reduced end-effector body
`e` and the maximal base body `b`:

* **Pose**: the welded frame on `e` must coincide with the welded frame on `b`.
* **Twist**: their spatial velocities at the shared boundary point must match.
* **Wrench**: the constraint wrench is exchanged equal-and-opposite (the boundary
  impulse `λ` and its reaction `−λ`).

Everything else (the arm's joints, the gripper's loops, gravity, actuation) is
handled *inside* each subsystem's own solver, untouched.

## 6. What is simplified (this PoC)

* **One** reduced subsystem, **one** maximal subsystem, **one** 6D boundary.
* **Staggered (Gauss–Seidel) coupling**, not a single monolithic solve. The
  reduced side is advanced first; the boundary impulse is computed from the
  reduced post-step twist and the maximal pre-step twist; the reaction is applied
  to both; then the maximal side is advanced. See §9.
* **Approximate maximal mass operator.** The boundary system uses the *base
  body's* free-body 6×6 spatial inertia `Mₘ` for the maximal Schur block. The
  true effective inertia at the base — accounting for the rest of the maximal
  subsystem and its loop constraints — is not exposed by Kamino. We deliberately
  **under-estimate** `Mₘ` (base body only) rather than over-estimate it: a
  rigid-composite estimate (summing the whole subsystem's inertia about the
  boundary as if it moved rigidly) *over-states* the effective inertia for a
  freely-articulating mechanism, which makes the staggered impulse overshoot and
  **diverge** (observed on the four-bar). Under-estimating under-corrects, which
  is the *stable* failure mode: the impulse is still real and fed back to both
  sides, and the residual is driven down over steps (and by position
  stabilization). The practical consequence is that the chosen base body should
  carry meaningful mass (the gripper's main `base` link, not a massless mounting
  frame). A configuration-aware effective inertia (e.g. an empirical probe of the
  maximal solver's boundary response) is the right longer-term fix — see §11 — and
  is now available as an opt-in (`Config.use_effective_inertia`, below).
* **Optional probed effective inertia (`use_effective_inertia`).** Instead of the
  free-base under-estimate, the maximal Schur block can be the *exact* effective
  inverse spatial inertia of the whole maximal subsystem at the base — loop
  closure, payload, and current contacts included — measured empirically once per
  frame. A separate cold-probe maximal solver applies six unit boundary impulses
  `e_k` at the base and reads the base-twist response, assembling the 6×6 operator
  `Aₘ_eff` (symmetrized, with an SPD fallback to the free-base block). On the
  Franka + 2F-85 example the probed angular block is ~3–4× larger than the
  free-base estimate (the root cause of the weak angular weld), so the impulse
  itself holds the weld: the angular pose error stays ~1.3° *even with the
  velocity-Baumgarte stabilization turned off*, versus needing the crutch before.
  The probe runs entirely on-device (its seven Kamino solves are wrapped in their
  own CUDA graph) for ~15–18 % added gripper cost. See `probe_maximal_inertia()`.
* **Clamped graceful degradation under dynamic overload.** Because `Mₘ`
  under-estimates the loaded effective inertia, the staggered impulse cannot hold
  the weld through a *dynamic* grasp (lifting/placing a payload, §10): the maximal
  side under-responds to the wrench the boundary solve predicts, so the Baumgarte
  bias winds up and ramps the boundary force, and near an arm singularity the
  reduced operational inverse-inertia amplifies even a small impulse into a
  joint-velocity whip on the arm. Three optional caps bound this so the failure
  mode stays *graceful under-correction* rather than a destructive transient: a
  maximal-side wrench cap (`max_boundary_force` / `max_boundary_torque`) bounds the
  force/torque exchanged at the seam — and, by Newton's third law, the reaction
  onto the arm — and a reduced-correction cap (`max_reduced_correction`) bounds the
  per-step arm velocity correction directly. On the Franka + 2F-85 cube-stacking
  example these cut the peak arm joint speed from ~25 to ~1 rad/s and the peak
  boundary force from ~310 to ~50 N while the weld stays sub-degree and
  sub-millimetre. They are caps, not a cure; the configuration-aware effective
  inertia of §11 is the real fix.
* **Velocity-level constraint with optional Baumgarte position stabilization.**
  The default is a small Baumgarte term; pure velocity-level coupling is the
  `baumgarte = 0` special case.
* **Boundary point = reduced end-effector COM.** The Jacobians are referenced
  there so that `Jᵀ` performs the correct wrench transport automatically (§7).
  The maximal base may sit at an arbitrary offset; the rigid offset is captured at
  construction time, so the two subsystems need not be co-located.
* **No contacts** between the two subsystems in the PoC (each subsystem still runs
  its own internal contacts/collisions if configured).
* **On-device 6×6 solve.** The coupling linear algebra (a 6×6 Schur solve plus an
  `ndof` Cholesky of the reduced mass matrix) runs in a single Warp kernel, so the
  coupled step performs no host round-trips and can be CUDA-graph captured together
  with the two child solvers. This is what makes the staggered scheme competitive:
  an un-captured step forces each child solver onto its slow host-synchronized
  iteration path (§12).

## 7. Mathematical formulation

### Coordinates and the boundary maps

Let the reduced generalized velocity be `v ∈ ℝⁿᵛ` and the maximal base body twist
be `V ∈ ℝ⁶` (Newton public convention: `(v_lin@COM, ω)`, world frame). Let `p*`
be the boundary point (the reduced EE COM).

* **Reduced boundary Jacobian** `Jᵣ ∈ ℝ⁶ˣⁿᵛ`: the rows of `newton.eval_jacobian`
  for body `e`, which satisfy `Jᵣ v = body_qd[e]` exactly (verified by Newton's
  own tests). Because the boundary point is the EE COM, no extra transport is
  needed.
* **Maximal boundary Jacobian** `Jₘ ∈ ℝ⁶ˣ⁶`: the rigid twist-transport (adjoint)
  from the base body COM to `p*`,

  ```
  Jₘ = [ I₃   −[r]ₓ ]      r = p* − p_baseCOM   (world)
       [ 0     I₃  ]
  ```

  so that `Jₘ V` is the base material velocity *at the boundary point*. (If `p*`
  is the base COM, `Jₘ = I₆`.)

### Constraint

Velocity-level rigid attachment at the boundary point:

```
C(v, V) = Jᵣ v − Jₘ V = 0
```

with the position residual `e_pose ∈ ℝ⁶` (translation error + axis-angle of the
relative rotation) used for Baumgarte stabilization.

### Impulse-level coupled (KKT) system

The quantity we want is the boundary impulse `λ ∈ ℝ⁶`. The minimal coupled system
that produces it is the saddle-point system from the prompt:

```
[ Mᵣ   0    −Jᵣᵀ ] [ Δv ]   [ 0  ]
[ 0    Mₘ    Jₘᵀ ] [ ΔV ] = [ 0  ]
[ Jᵣ  −Jₘ    0   ] [ λ  ]   [ r_c ]
```

with

* `Mᵣ ∈ ℝⁿᵛˣⁿᵛ` the **exact** reduced mass matrix (`newton.eval_mass_matrix`),
* `Mₘ ∈ ℝ⁶ˣ⁶` the base body's spatial inertia `diag(m I₃, R Iₗₒ𝒸 Rᵀ)` (approx, §6),
* `r_c = c_target − c₀`, the residual that the corrected velocities must achieve:
  `c₀ = Jᵣ v⁺ − Jₘ V₀` (current boundary velocity mismatch) and
  `c_target = −(β/dt) · e_pose` (Baumgarte target relative velocity).

### Schur complement (what the code actually solves)

Eliminating `Δv = Mᵣ⁻¹ Jᵣᵀ λ` and `ΔV = −Mₘ⁻¹ Jₘᵀ λ` collapses the system to a
6×6 solve in `λ`:

```
A λ = r_c ,    A = Jᵣ Mᵣ⁻¹ Jᵣᵀ + Jₘ Mₘ⁻¹ Jₘᵀ  (+ εI)
```

`A` is the **boundary inverse-inertia** (Delassus / "operational-space inverse
mass") of the two subsystems seen at the boundary. A small `ε` regularizes
near-singular configurations. This is the same structure used for contact and
articulation-coupling impulse solves throughout rigid-body dynamics; here the two
"bodies" are *a whole reduced subsystem* and *a whole maximal subsystem*.

## 8. How the boundary impulse maps back to each side

Once `λ` is known:

* **Reduced side** — generalized impulse `τ_boundary = Jᵣᵀ λ`, applied as a
  velocity correction `Δv = Mᵣ⁻¹ Jᵣᵀ λ` to the arm's `joint_qd`. (Because the
  reduced solver has already advanced this step, the impulse is applied as a
  velocity projection; an equivalent pre-step `control.joint_f += τ_boundary / dt`
  formulation is noted in the code.)
* **Maximal side** — spatial impulse `f_boundary = −Jₘᵀ λ` at the base COM,
  injected as a wrench `f_boundary / dt` into `state.body_f[b]` and integrated by
  Kamino's own constraint solver. `Jₘᵀ` performs the boundary→COM wrench transport
  (`τ_COM = τ* + r × f`) automatically, so no manual moment shift is needed.

`+Jᵣᵀλ` to one side and `−Jₘᵀλ` to the other is exactly Newton's third law across
the seam: the same `λ` produces equal-and-opposite generalized/Cartesian reactions.

## 9. How this differs from naive operator splitting / one-way coupling

A **naive one-way** scheme (the anti-pattern the prompt warns against) would, each
step, copy the arm EE pose onto the gripper base and let the gripper follow. That
is kinematic teleoperation: the gripper feels the arm, but the arm never feels the
gripper's inertia or reaction. Momentum is not conserved; a heavy gripper would be
dragged weightlessly.

`SolverBoundaryImpulse` instead computes a single impulse `λ` from the *coupled*
inverse-inertia `A` of **both** subsystems and applies the reaction to **both**:

* the arm is decelerated/accelerated by `Δv = Mᵣ⁻¹Jᵣᵀλ` — it *feels* the gripper;
* the gripper is pushed by `−Jₘᵀλ` — it *feels* the arm.

It is a staggered (Gauss–Seidel) approximation of the monolithic solve, not a
master/slave copy. The distinction is the whole point of the solver.

### Per-step sequence (as implemented)

1. Advance the reduced subsystem with its own solver → `q⁺, v⁺`, EE pose/twist.
2. Read reduced boundary pose `Tₑ`, twist `Vₑ = Jᵣ v⁺`, build `Jᵣ`, `Mᵣ`.
3. Read maximal boundary pose `T_b`, twist `V_b = Jₘ V₀`, build `Jₘ`, `Mₘ`
   (from the maximal *input* state, pre-advance).
4. Solve `A λ = r_c` (Schur, §7).
5. Apply `Δv = Mᵣ⁻¹Jᵣᵀλ` to the reduced output velocity.
6. Apply `f_boundary = −Jₘᵀλ` as a wrench into the maximal input `body_f[b]`.
7. Advance the maximal subsystem with its own solver (now feeling `f_boundary`).
8. Log boundary pose error, velocity error (pre and post correction), and `λ`.

## 10. What is *not* implemented yet

* A **monolithic** simultaneous solve of arm + gripper (§11).
* **Exact** maximal-side effective inertia (the loop-closure-aware Schur block);
  the PoC uses the free-body base inertia.
* **Symmetric same-instant** coupling: the reduced side is corrected after its
  advance, the maximal side before its advance — an O(dt) staggering artifact.
* **Multiple** boundaries, multiple subsystems, or a boundary that is itself a
  compliant/contact interface.
* A **dynamic grasp** on the real robot. The Franka + 2F-85 benchmark (§12) runs
  a *static hold*; driving the gripper closed consistently across all three
  solvers is left as future work.

## 11. What a truly monolithic full-system solve would require

A monolithic solver would assemble and solve the *combined* KKT system once per
step instead of staggering:

* A **shared linear system** over `(Δv, ΔV, λ)` (and, for the maximal side, its
  internal loop-constraint multipliers) advanced simultaneously — i.e. the
  maximal solver's own constraint solve and the boundary constraint solved in one
  fixed-point/Newton iteration, not sequentially.
* Access to each subsystem's **unconstrained step operator and its effective
  inverse-inertia** at the boundary (the true `Mₘ⁻¹` including loop closure, and
  `Mᵣ⁻¹` including the arm's actuation/damping), which today are private to the
  solvers. The smallest enabling change is a per-solver
  `apply_boundary_impulse(body, λ) → ΔV_at_boundary` / `effective_inverse_inertia(body)`
  hook that the coupler can call inside an iteration loop.
* **Position-level (index-3) stabilization** or a proper constraint manifold
  projection, replacing the Baumgarte velocity target, for exact pose tracking.
* Consistent **same-instant** velocities for both sides (advance both
  unconstrained, then project), removing the staggering artifact of §10.

The present solver is deliberately the smallest faithful step toward that: it
computes and exchanges a real boundary impulse today, and exposes exactly the
seam where the monolithic iteration would later be inserted.

## 12. Benchmark: Franka FR3 + Robotiq 2F-85

`examples/experimental/example_franka_2f85_benchmark.py` scales the PoC from the
2R-arm + four-bar toy to the real target system and answers the question *is the
hybrid faster or higher quality than running everything in one solver?* It builds
the **Franka FR3** arm (7-DOF, from the cached `fr3.urdf`) and the **Robotiq
2F-85** gripper (14-body closed chain, from the MuJoCo-Menagerie `2f85.xml`; the
two `connect` equalities import as Newton ball loop joints), and compares three
configurations on the **identical** static-hold scenario (arm PD-holds a ready
pose, gripper held, gravity on, `dt = 1e-3`, GPU):

* **HYBRID** — Franka in `SolverFeatherstone` + 2F-85 in `SolverKamino`, coupled
  by `SolverBoundaryImpulse` (flange `fr3_link8` welded to the gripper `base`).
* **FULL-KAMINO** — the whole arm+gripper (24 bodies) in `SolverKamino`.
* **FULL-MUJOCO** — the whole arm+gripper in `SolverMuJoCo` (loops as equalities).

(There is no FULL-FEATHERSTONE: `SolverFeatherstone` cannot represent the 2F-85
closed loop, so it would silently simulate a different, open-chain robot — which
is exactly *why* a hybrid is needed to pair a reduced arm with a closed-chain
gripper.)

### Measured results (NVIDIA TITAN RTX, 200 timed steps after warm-up)

All three configs are **CUDA-graph captured** (the boundary coupling now runs
on-device, so the hybrid step captures just like the two baselines — a
prerequisite for a fair comparison; see below).

| Config | ms / step | bodies | weld pose error | stable |
|---|---:|---:|---:|:--:|
| FULL-MUJOCO | **0.77** | 24 | — | ✓ |
| FULL-KAMINO | 27.8 | 24 | — | ✓ |
| HYBRID | **13.0** | 24 | 1.3·10⁻⁴ m | ✓ |

HYBRID per-component breakdown [ms/step]: arm (Featherstone) **0.97**,
gripper (Kamino, 14-body) **11.4**, on-device coupling **0.67**.

### Why graph capture is the whole story

`SolverKamino` runs its PADMM iteration to convergence with a device-side
`wp.capture_while` loop. **Un-captured**, each iteration's residual check
round-trips to the host, so the per-step launch + synchronization overhead
dominates the actual compute by ~10×. **Captured**, the loop runs entirely on
the GPU. Measured on the isolated gripper, capture turns a 150 ms/step solve into
a 13.8 ms/step solve — a **~11× speedup with no change in the math.**

The previous version of this note assembled the boundary system in NumPy on the
host. That host code sat *inside* the coupled step, which made the whole hybrid
step impossible to graph-capture while both *baselines were captured* — an
apples-to-oranges comparison that made the hybrid look ~9× slower than it is.
Moving the 6×6 Schur solve into a Warp kernel removed the only host round-trip in
the step, so all three configs are now captured on equal footing.

### What the numbers say

* **The hybrid is correct.** It is stable and holds the rigid weld between the
  reduced end-effector and the maximal gripper base to **sub-millimeter**
  (pose error ≈ 1.3·10⁻⁴ m) — the coupling does what it claims.
* **The hybrid beats full-Kamino.** At 13.0 ms/step it is **~2.1× faster** than
  full-Kamino (27.8 ms) on the same system.
* **Splitting the maximal subsystem off *does* shrink its cost.** Kamino on the
  *isolated 14-body gripper* (11.4 ms, captured) is **cheaper** than Kamino on the
  *full 24-body* arm+gripper (27.8 ms): the hybrid gives the maximal solver a
  smaller problem and pays only a cheap reduced-arm solve (≈ 1 ms) plus negligible
  coupling (≈ 0.7 ms) for the rest. *(This reverses an earlier finding that was an
  artifact of timing the gripper solve un-captured while the full system was
  captured.)*
* **The coupling is cheap.** The on-device boundary layer (eval_jacobian /
  eval_mass_matrix + the 6×6 Schur solve kernels) costs only ~0.7 ms/step.
* **For raw speed, a single mature reduced solver still wins decisively.**
  `SolverMuJoCo` represents the same closed loop via equality constraints and is
  ~36× faster than full-Kamino and ~17× faster than the hybrid.

### Verdict and caveats

The hybrid's value on this system is **modularity and correctness** plus a real
**speedup over the full maximal-coordinate solver**: it lets a reduced-coordinate
arm engine that *cannot* represent loops still drive a true maximal-coordinate
closed-chain gripper, with a faithful rigid interface, at ~2× the speed of running
the whole robot in Kamino. It does not beat a single mature reduced solver
(`SolverMuJoCo`) that can model the loop directly; the hybrid earns its keep only
when the closed-chain subsystem genuinely needs a maximal-coordinate solver.
Caveats that make the comparison fair and reproducible: all three configs are
CUDA-graph captured after an identical uncaptured warm-up to the held steady
state; the Franka needs small **joint armature** to be stable under
`SolverFeatherstone`'s semi-implicit integration at `dt = 1e-3` (the URDF carries
none); the scenario is a **static hold** (a dynamic grasp is future work, §10);
and all three configs use identical models, gains, armature, and `dt`.
