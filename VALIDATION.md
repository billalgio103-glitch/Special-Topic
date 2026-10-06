# Execution record

Preparation environment: Python 3.12.14 on Linux.

Completed:

- Retrieved source metadata and verified 465 EMPS, 407 HRTEM and 256 cobalt records.
- Downloaded the two cobalt HDF5 files and verified their published MD5 hashes.
- Parsed one real image/label pair from each source into SQLite.
- Verified that three-source subset: primary/foreign keys, raw and derived
  checksums, target dimensions, binary values, pixel counts and group isolation.
- Built the subset twice: both runs contained 3 images and 29 measured objects,
  with logical fingerprint
  `7cac16aaa748ccb3032a331ddd49bc0bf4e8382aa4f8d56116bf67b9f35f52f9`.
- Inspected native-image/label overlays for those three examples.
- Six semantic tests passed (touching instances, binary components,
  anisotropic calibration, border exclusion, missing scale, sparse IDs and
  grouped split determinism).

Pending at this checkpoint:

- Full-source download and 1,128-image database validation.
- Full-source idempotency check.
- Podman execution of the default `just` recipe. Podman is not installed in
  the preparation environment; native Python tests do not replace this gate.

This is an honest execution record; the subset is not a fully populated
assignment database. The default recipe requires all three complete sources.
