# PhysTwin 3D contracts, version 1

This directory contains the first project-owned 3D contract examples.

The contract version is independent from the product generation. The existing
2D `tracking.json` and `reconstruction.json` files remain legacy V1 contracts.
They are not moved or changed during P0.

## Files

- `examples/scene_observation.json` is reconstructed visual evidence.
- `examples/physical_scene.json` is a draft physical interpretation.
- `examples/physical_scene_tether.json` is the first executable P4 scene.
- `examples/da3_w2c_fixture.json` locks the DA3 OpenCV `w2c` to observation-world conversion.
- `examples/tram_c2w_fixture.json` locks the TRAM OpenCV `c2w` plus camera-space SMPL24 joints to observation-world conversion.
- `vision/reconstruction/contracts.py` contains the executable contract validation.
- Body evidence is namespaced as `extensions.phystwin.humans.v1`. It does not change core coordinates.
- `phystwin.reconstruction_evaluation` records P3 metrics, alignment, coverage,
  benchmark provenance, and artifact hashes. It stays separate from
  `SceneObservation` and `PhysicalScene`.
- `phystwin.simulated_world_state` records P4 simulator output. Three.js reads
  this project-owned rollout instead of Newton objects.

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

P4 defines one concrete executable payload:

- one `rigid_body` with a sphere shape, `T_world_body_initial`, mass, and
  initial linear and angular velocity
- one `distance` constraint with a world anchor, body-local attachment, and
  fixed rest length
- Newton XPBD execution settings with duration and fixed step

The original example stays draft. The P4 tether fixture is standalone and
already uses SI units, so its observation alignment fields are null and
`scale_source` is `not_applicable`.

## `SimulatedWorldState`

The P4 output schema is `phystwin.simulated_world_state`, version 1. It stores:

- the source `PhysicalScene` ID and canonical SHA-256
- Newton, Warp, solver, device, and CUDA identity
- the fixed-step timeline
- row-major `T_world_body` samples and body velocities
- constraint geometry needed by the viewer
- runtime and GPU-memory metadata
- finite, time, gravity, tether-error, XYZ-motion, and repeat-run checks
- warnings and failures

## Extension rule

Future evidence that is not part of the P0 core lives under a versioned,
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
.\.venv\Scripts\python.exe -m unittest vision.test_reconstruction_contracts vision.test_reconstruction_p1 vision.test_reconstruction_p2 vision.test_reconstruction_p3 physics3d.test_p4 -v
.\.venv-physics\Scripts\python.exe -m unittest physics3d.test_p4 -v
```
