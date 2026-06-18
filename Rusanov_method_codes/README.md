Run `moving-piston-no-left-gas.ipynb`. You will need Julia, with the `IJulia` package to run the Jupyter notebook. Note: The data is already generated. Just plot with the python file.

Then run `python3 plot_waterfall.py`.
Note that the python plot may take a while (upto a minute).

## Equations solved

The notebook solves the 1D compressible **Navier–Stokes** equations (a
simplified form, with a single scalar momentum diffusivity and no heat
conduction) for a gas pushed by a moving piston into an initially
evacuated region, under an external harmonic-trap potential Φ(x,t). In
conservative form, with state vector `U = [ρ, ρu, E]`:

```
∂U/∂t + ∂F(U)/∂x = S(U, x, t) + V(U)

U = [ρ, ρu, E]ᵗ
F = [ρu, ρu² + pressureScale·p, u(E + pressureScale·p)]ᵗ
S = [0, -ρ·∂Φ/∂x, -ρu·∂Φ/∂x]ᵗ
V = [0, ν·∂²u/∂x², ∂/∂x(ν·u·∂u/∂x)]ᵗ,   ν = nuMom = 1/ReynoldsNumber
```

with the ideal-gas closure `p = (γ-1)(E - ½ρu²)/pressureScale` and sound
speed `c = sqrt(γ·pressureScale·p/ρ)`. The `pressureScale` factor is a
nondimensionalization prefactor (set from the Mach number) so that the
pressure-gradient and trap-force terms are consistent in code units.

The viscous term `V` is real physics here, not a numerical fix: it's
parameterized directly by the physical `ReynoldsNumber` (`nuMom =
1/ReynoldsNumber`), and the CSV exports / different `Re` runs in this
folder (Re=50,100,500,1000,5000) sweep this physical viscosity to see its
effect on the flow. It's the momentum-diffusion (`ν·∂²u/∂x²`) and
viscous-work (`∂/∂x(ν·u·u_x)`) terms of the compressible Navier–Stokes
equations, simplified to a single coefficient and with no separate
heat-conduction term — it is independent of, and in addition to, the
numerical viscosity introduced by the Rusanov flux described next.

The left boundary is a moving wall (the piston, prescribed by `Upiston(t)`)
implemented via a reflecting ghost cell mirrored about the piston speed;
the right boundary is transmissive (outflow).

## Temperature: set only as an initial condition, not enforced — no heat equation

`temp` (set to `1`) is **not** held fixed during the run; it only shapes
the `t=0` state. It is used solely to build the nondimensionalization
(`c0 = sqrt(γ·temp)`, `mach = 1/c0`, `pressureScale = 1/(γ·mach²)`) and the
initial profiles:

```
InitialDensityProfile(x)  ∝ exp(-Φ(x,0) / (pressureScale·temp))
InitialPressureProfile(x) = temp · exp(-Φ(x,0) / (pressureScale·temp))
```

i.e. the classic isothermal hydrostatic equilibrium `ρ ∝ exp(-Φ/T)`,
`p = T·ρ`, for a gas sitting in the trap at temperature `temp` before the
piston starts moving. `temp` never appears again in `prim`, `flux`,
`aSound`, `rusanov`, `sourceVec`, `fvStep`, or `dtFromCFL`. Once the
simulation starts, `ρ`, `ρu`, `E` evolve as independent conserved
quantities, so the implied temperature `T = p/ρ` is a free, derived field
that responds to compression, expansion, and viscous heating — nothing in
the code relaxes it back to `temp` or to any other target.

**No heat equation is solved.** The energy equation's only terms are the
advective/pressure-work flux `u(E + pressureScale·p)`, the trap-force work
source `-ρu·∂Φ/∂x`, and the viscous-work flux divergence
`∂/∂x(ν·u·∂u/∂x)` from `viscousEnergyTerm` — there is no Fourier
conduction term `κ·∂²T/∂x²`. The code documents this directly with the
comment *"No heat-conduction term here; this is only viscous work flux
divergence."* So viscous heating can raise `T` locally where there's
shear, but that heat is never diffused away — the model is adiabatic
compressible flow with viscous *momentum* dissipation, not a coupled
flow + heat-conduction system.

## Energy flow, temperature changes, and what's missing: thermal conduction

There is no heat equation, but `T` is still fully dynamic — it changes
because internal energy is part of the conserved total energy
`E = pressureScale·p/(γ-1) + ½ρu²`, and several terms move energy into or
out of that internal-energy slot:

1. **Adiabatic compression/expansion (pdV work)** — the energy flux
   `u(E + pressureScale·p)` contains a pressure-work term. As the piston
   compresses gas, that term pumps kinetic energy into internal energy
   faster than pure advection would, raising `p` and hence `T` — ordinary
   adiabatic heating, with no conduction involved.
2. **Shock heating (irreversible, via the Rusanov dissipation)** — across
   a shock, kinetic energy converts to internal energy (entropy
   production), as in a Rankine–Hugoniot jump. The Rusanov flux's
   `−½·aMax·(U_R−U_L)` term acts on all three conserved variables,
   including `E`, so the captured shock correctly raises `T` behind the
   front, standing in for the real entropy-producing process at a shock.
3. **Explicit viscous heating** — `nuMom` doesn't just diffuse momentum;
   `viscousEnergyTerm` adds the viscous-work flux `∂/∂x(ν·u·u_x)` to the
   energy equation. Decomposing `E` into kinetic + internal energy via the
   momentum equation reveals a genuine local heating source proportional
   to `ν·(∂u/∂x)²` — shear literally generates heat, like friction.
4. **Trap-force work** — `-ρu·∂Φ/∂x` exchanges energy between the gas and
   the external potential, changing bulk kinetic energy, which later
   shows up as a `T` change once that kinetic energy is thermalized via
   (2) or (3).

**What's missing is thermal conduction (Fourier's law).** A full
compressible Navier–Stokes–Fourier system would add a heat flux
`q = -κ·∂T/∂x` and a term `+κ·∂²T/∂x²` to the energy equation; this is
absent here. Consequences:

- **No molecular smoothing of temperature gradients.** Heat generated by
  shocks or viscous shear can only move by advection (gas physically
  carrying it) — it cannot diffuse into neighboring cooler gas the way
  real heat conduction would. A temperature spike next to cold gas stays
  sharp indefinitely in this model, where a real gas would smooth it out
  over a conductive timescale.
- **Implicit Prandtl number → ∞.** Real gases have momentum and heat
  diffusing at comparable rates (`Pr = ν·cp/κ ~ O(1)`, e.g. air ≈ 0.7).
  Setting `κ = 0` while keeping `ν = 1/Re` finite is an extreme,
  nonphysical limit, not a parameter choice with a small effect.
- **A small amount of *numerical* heat conduction does sneak in.** Because
  the Rusanov flux's dissipative term acts on the `E` component too (not
  just momentum), it adds an `O(Δx)` numerical diffusion to the
  energy/temperature field as a truncation-error artifact. This shrinks
  under mesh refinement and is not physical conductivity — it shouldn't be
  relied on to represent real thermal smoothing.

Net effect: this is an adiabatic-plus-viscous model (heat is generated and
advected, never conducted) — well suited to studying shock structure and
bulk viscous dynamics, but it will systematically keep temperature
gradients/spikes sharper and longer-lived than a real gas with finite `κ`
would.

## How the Rusanov method handles discontinuities

The convective part of these equations (i.e. with the viscosity ν set to
zero, leaving just the Euler equations) is hyperbolic and develops
genuine discontinuities
(shocks, contact surfaces) even from smooth initial data — here, the
piston compresses the gas and launches a shock ahead of it. The `Re=5000`
default case is only mildly viscous, so this shock-formation behavior is
still essentially Euler-like and the discontinuity is sharp on the grid
scale. A naive
centered-difference discretization is unstable at such discontinuities, so
the code uses a finite-volume scheme with the Rusanov (local
Lax–Friedrichs) numerical flux at every cell interface:

```
F_{i+1/2} = ½(F_L + F_R) − ½·aMax·(U_R − U_L),   aMax = max(|u_L|+c_L, |u_R|+c_R)
```

The first term is the centered average flux; the second term is a
diffusive correction proportional to the jump in the conserved variables
across the interface, scaled by the fastest local wave speed `aMax`
(velocity plus sound speed on either side). This is what stabilizes the
scheme: near a shock or contact discontinuity, `U_R − U_L` is large, so the
correction injects just enough numerical dissipation to keep the update
bounded and convergent (no oscillatory blow-up), while away from
discontinuities the jump is small and the flux reduces to the centered
(non-dissipative) approximation.

Formally, this diffusive term is equivalent (via Taylor expansion of the
flux difference) to adding an artificial viscosity of size `½·aMax·Δx` to
the underlying PDE. It is first-order in the grid spacing `Δx`, so it
vanishes under mesh refinement, but it is exactly what prevents the
divergence/blow-up that a purely centered flux would produce at a shock.

## Time-step selection (CFL condition)

`dtFromCFL(U, t)` picks the largest stable timestep by computing three
independent stability limits and taking the minimum:

1. **Advective (hyperbolic) CFL**: `dt_adv = CFL · Δx / s_max`, where
   `s_max = max(|u| + c)` over all active gas cells. This bounds how far
   the fastest wave (advection + sound speed) can travel in one step to
   less than `CFL` (= 0.45) cell widths, which is required for the
   explicit Rusanov update to remain stable.
2. **Diffusive (viscous) CFL**: `dt_diff = CFLvisc · Δx² / ν_eff,max`,
   where `ν_eff = nuMom/ρ` is the local effective momentum diffusivity.
   Explicit treatment of the `ν·∂²u/∂x²` term requires the parabolic
   condition `dt ∝ Δx²` rather than `Δx`, with safety factor
   `CFLvisc` (= 0.2).
3. **Source-term (forcing) CFL**: `dt_force = CFLforce · s_max / g_max`,
   where `g_max = max|∂Φ/∂x|` is the largest trap acceleration. This
   keeps the harmonic-trap source term from changing the velocity by more
   than a fraction (`CFLforce` = 0.25) of the local wave speed in a single
   step.

The actual step is `dt = min(dt_adv, dt_diff, dt_force, tEnd - t)`, so
whichever physical process (advection, viscous diffusion, or external
forcing) is locally most restrictive sets the timestep. Cells inside the
piston-evacuated region (`x < xpiston(t)`) are excluded from these
estimates since they hold no real gas.