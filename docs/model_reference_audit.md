# TokaSys Model Reference Audit

This document records the reference lineage and remaining reduced-model
assumptions for the current differentiable 0D/profile tokamak system code.

## Plasma

- D-T fusion reactivity: Bosch and Hale, Nuclear Fusion 32, 611-631 (1992).
  Status: reference-backed analytic fit.
- Fusion power density: standard 50:50 D-T mixture formula integrated over the
  modeled density/temperature profiles. Status: reference-backed, profile aware.
- Bremsstrahlung: nonrelativistic electron-ion engineering form proportional to
  `Z_eff n20^2 sqrt(T_keV)`. Status: reduced but standard 0D/profile model.
- Synchrotron radiation: Albajar/Johner/Granata global fit with wall
  reflectivity and optical-depth-like attenuation. Status: reference-backed
  global fit, not a ray-tracing calculation.
- Resistivity and ohmic power: Spitzer-Harm resistivity with profile-integrated
  `eta * j_ind * j_tot`. Status: reference-backed reduced transport closure.
- Confinement: IPB98(y,2) H-mode scaling with full-space or self-consistent
  energy closure. Status: empirical system-code model.
- Pedestal/profile model: TokaGrad-style alpha-critical pedestal plus stored
  energy closure, with PROCESS-style profile options for validation. Status:
  reduced; no EPED/peeling-ballooning stability solve.
- Bootstrap current: Sauter-inspired collisionality/pressure-gradient closure.
  Status: reduced; not a full neoclassical local coefficient implementation.
- Current drive: normalized-efficiency closure following current-drive systems
  studies. Status: proxy; no wave/beam deposition or Fokker-Planck model.
- Fast ions: slowing-down stored-energy estimate plus IPDG89/PROCESS-like
  alpha-beta option. Status: reduced; no orbit/transport or anisotropy model.

## Geometry

- Plasma volume and profile integration: elongated axisymmetric volume elements.
  Status: reference-backed system-code geometry.
- Surface area split: Miller-like boundary quadrature used to distinguish
  inboard/outboard first-wall area. Status: reduced but differentiable.
- Radial build: explicit inboard/outboard ledger. Status: accounting model;
  no CAD envelope, ports, maintenance sectors or cryostat geometry.

## Magnets

- TF peak field, current density and stress: magnetic-pressure and winding-pack
  packing surrogates. Status: reduced; no detailed structural FEA, quench,
  joints, grading, insulation or support-shell model.
- Superconductor Jc(B,T,strain): reduced engineering derating curves inspired
  by practical critical-surface fitting. Status: proxy; not vendor-specific
  REBCO/Nb3Sn strand/tape qualification.
- CS flux swing: startup, flat-top volt-second and peak-field/current-density
  constraints. Status: reduced; no detailed inductive circuit/equilibrium solve.
- PF coils: fixed differentiable coil placement with coil-by-coil J/B/stress.
  Status: proxy; no free-boundary control-coil optimization.

## Nuclear Island

- Blanket concepts: concept-dependent TBR, attenuation, energy multiplication,
  pumping and tritium inventory coefficients. Status: reduced surrogate.
- TF fluence/dpa: attenuated neutron wall-loading proxy. Status: proxy; not
  MCNP/FISPACT or activation-informed lifetime.
- Tritium doubling time: inventory/residence-time balance. Status: reduced fuel
  cycle metric.
- First-wall heat load: surface-average thermal loading. Status: reduced; FW
  temperature, coolant channel geometry and material stress are not explicit.

## Exhaust

- SOL width and divertor heat flux: Eich-like Bp scaling with flux expansion and
  spreading-factor reduction. Status: reference-backed scaling surrogate.
- Divertor target temperature/lifetime: heat-flux utilization model. Status:
  proxy; no detached plasma, target material erosion or cooling-channel solve.

## Plant And Economics

- Thermal/electric balance: PROCESS-style system-code accounting with separate
  auxiliary, current-drive, coolant pumping, vacuum, tritium and cryogenic loads.
  Status: reduced accounting model.
- Availability: component-lifetime/RAMI-inspired reduced model. Status: proxy;
  no stochastic failure/repair simulation.
- Cost model 0: parametric scaling with component volumes, stored energy and
  wall-plug systems. Status: calibrated surrogate; not a bottom-up cost model.
- Cost model 1: whole-plant `C/C_ref = (P/P_ref)^0.6` capacity scaling,
  anchored to the model-0 component base costs at 1000 MW gross electric
  capacity. Status: standard conceptual economies-of-scale law.
- Cost model 2: Jo et al., *Energies* 14, 6817 (2021), Section 2.3,
  Equations (8)-(14) and Tables 2-3
  (https://doi.org/10.3390/en14206817). Published unit costs and plant-level
  equations are used directly in constant 2010 USD. The paper obtained
  component length, mass and fusion-island volume from a separate coupled
  systems analysis; TokaSys reconstructs those inputs from its D-coil
  perimeter, a documented 50 kA conductor assumption, component volumes,
  effective material densities and a cylindrical fusion-island envelope.
  As in the paper, the blanket material costing assumes a PbLi/FMS/SiC
  blanket, independently of TokaSys' blanket-concept selector.
  Status: reference-backed cost equations with explicit geometry/material
  adapters, not a construction-grade bottom-up estimate.

## Validation

- PROCESS reference comparisons use mapped regression cases from the UKAEA
  PROCESS repository. Status: useful for system-code consistency checks, but not
  a one-to-one replacement for PROCESS' full physics and engineering stack.
