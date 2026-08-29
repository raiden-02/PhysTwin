# PhysTwin 3D contracts, version 1

This directory contains the first project-owned 3D contract examples.

The contract version is independent from the product generation. The existing
2D `tracking.json` and `reconstruction.json` files remain legacy V1 contracts.
They stay on the image-space path.

## Files

- `examples/scene_observation.json` is reconstructed visual evidence.
- `examples/physical_scene.json` is a draft physical interpretation.
- `examples/physical_scene_tether.json` is the first executable tether scene.
- `examples/physical_scene_tether_fit_template.json` is the bounded tether-fit template.
- `examples/physical_motion_observation_tether_synthetic.json` is a compact
  sample of the generated metric target.
- `examples/inverse_physics_fit_blocked_input.json` records the strict gate
  result for the current relative-scale observation.
- `examples/da3_w2c_fixture.json` locks the DA3 OpenCV `w2c` to observation-world conversion.
- `examples/tram_c2w_fixture.json` locks the TRAM OpenCV `c2w` plus camera-space SMPL24 joints to observation-world conversion.
- `vision/reconstruction/contracts.py` contains the executable contract validation.
- Body evidence is namespaced as `extensions.phystwin.humans.v1`. It does not change core coordinates.
- `phystwin.reconstruction_evaluation` records reconstruction metrics, alignment, coverage,
  benchmark provenance, and artifact hashes. It stays separate from
  `SceneObservation` and `PhysicalScene`.
- `phystwin.simulated_world_state` records simulator output. Three.js reads
  this project-owned rollout instead of Newton objects.
- `phystwin.physical_motion_observation` stores metric 3D body-origin evidence
  for the inverse-physics objective.
- `phystwin.inverse_physics_fit` stores fit status, bounds, fitted values,
  objective metrics, artifact hashes, and validation.

Large geometry, masks, depth arrays, and model-native outputs are referenced as
artifacts. They are not embedded in these JSON envelopes.

## `SceneObservation`

Required core fields:

- `schema`: `phystwin.scene_observation`
- `version`: `1`
- `observation_id`
- `timeline`: ordered samples with original source frame numbers
- `coordinates`: canonical world, camera, image, transform, and scale rules
- `sources`: input video identities
- `artifacts`: external output identities
- `cameras`: intrinsics and `T_world_camera` samples
- `static_scene`: geometry artifact references
- `provenance`: adapter and estimator identity
- `extensions`: future namespaced evidence

The document says what was reconstructed. It does not contain a selected
physics model.

## `PhysicalScene`

Required core fields:

- `schema`: `phystwin.physical_scene`
- `version`: `1`
- `scene_id`
- canonical physical coordinates and SI units
- `observation_alignment`
- `execution`
- `world`
- `model` component containers
- `parameters`
- `provenance`
- `extensions`

The example remains a draft because scale, alignment, and a simulator backend
are unresolved. Draft serialization is valid. Execution is not.

The first executable payload is:

- one `rigid_body` with a sphere shape, `T_world_body_initial`, mass, and
  initial linear and angular velocity
- one `distance` constraint with a world anchor, body-local attachment, and
  fixed rest length
- Newton XPBD execution settings with duration and fixed step

The original example stays draft. The tether fixture is standalone and
already uses SI units, so its observation alignment fields are null and
`scale_source` is `not_applicable`.

## `SimulatedWorldState`

The rollout schema is `phystwin.simulated_world_state`, version 1. It stores:

- the source `PhysicalScene` ID and canonical SHA-256
- Newton, Warp, solver, device, and CUDA identity
- the fixed-step timeline
- row-major `T_world_body` samples and body velocities
- constraint geometry needed by the viewer
- runtime and GPU-memory metadata
- finite, time, gravity, tether-error, XYZ-motion, and repeat-run checks
- warnings and failures

## Inverse-physics contracts

`PhysicalMotionObservation` is not a second `SceneObservation`. It is the small
metric signal consumed by one inverse-physics objective. It can be generated
from a synthetic rollout, an eligible `humans.v1` pelvis track, or an
`entities.v1` object root.

The human adapter still requires `metric_measured` scale and measured
alignment. The entity adapter requires known-distance scale plus an explicit
physical-up declaration (`level_camera` assumed, or a supplied vector).
First-camera `+Y` is not treated as measured gravity. Guessed or estimator
scales stay `BLOCKED_INPUT`.

`InversePhysicsFit.validation` stores `execution_valid` and `quality`.
`quality.status` is `unassessed` on real fits and `synthetic_checked` on
synthetic recovery. Real `validation.passed` means the Newton run executed.
It does not grade residual quality.

`InversePhysicsFit` records the same three parameters. The length-fitting
profile is `tether_length_initial_tangent_velocity_v1`. When tether length
established scale, use
`tether_initial_tangent_velocity_fixed_length_v1` and set
`rest_length_m.held_fixed` to true. Status is `COMPLETE`, `BLOCKED_INPUT`, or
`FAILED`.

See [inverse-physics.md](../../../docs/inverse-physics.md) for the
objective, bounds, gate, optimizer, and failure behavior.

## Extension rule

Evidence that is not part of the core envelope lives under a versioned,
namespaced key in `extensions`, for example:

```json
{
  "extensions": {
    "phystwin.entities.v1": {},
    "phystwin.humans.v1": {},
    "phystwin.image_evidence.v1": {},
    "phystwin.uncertainty.v1": {}
  }
}
```

Readers may preserve unknown extensions. Unknown extensions cannot change the
meaning of core coordinates, time, units, transforms, or scale.

See [the 3D architecture document](../../../docs/architecture-3d.md) for the
full coordinate and migration decisions.

Run the contract checks from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest vision.test_reconstruction_contracts vision.test_reconstruction_p1 vision.test_reconstruction_p2 vision.test_reconstruction_p3 -v
.\.venv-physics\Scripts\python.exe -m unittest physics3d.test_p4 physics3d.test_p5 -v
```
