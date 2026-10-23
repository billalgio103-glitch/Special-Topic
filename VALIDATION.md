# Execution record

Validated on 2026-10-06 using Python 3.12.14 on Linux.

## Full-source execution

The full database was built from all 1,746 locked raw assets (approximately
28.3 GB), with no synthetic data or omitted records:

| Source | Images |
| --- | ---: |
| EMPS | 465 |
| HRTEM | 407 |
| Co3O4 | 256 |
| Total | 1,128 |

The database contains 23,386 measured objects. These are EMPS particle
instances and HRTEM/Co3O4 connected components; the total is not a count of
individually resolved physical particles across all sources.

Completed checks:

- Full raw-asset checksums and expected source counts.
- SQLite integrity, foreign keys and paired-image references.
- Derived mask checksums, dimensions, binary values and foreground pixel totals.
- Group isolation and exact-image leakage between training, validation and test.
- Eight scientific and atomic-publication unit tests.
- Dependency consistency (`python -m pip check`).
- Visual inspection of generated dataset overview, mask overlays and pixel-size
  distributions.

The machine-readable report is
[`validation/native-validation.json`](validation/native-validation.json).

## Full rebuild / idempotency

The validated full database was completely rebuilt from the same locked raw
assets. Both builds contained 1,128 images and 23,386 objects with identical
sorted logical table contents. The rebuilt outputs then passed full verification.

Logical database fingerprint (SHA-256):

```text
cab870d82508ffcb3589248f91797cef2ab4e00fbb51bddf8413da16d310c5a3
```

See [`validation/native-idempotency.json`](validation/native-idempotency.json).
This comparison checks database contents, rather than SQLite file-layout bytes.
To repeat the two-build check after downloading the data, run `just idempotency`.

## Container entry point

The preparation environment cannot run Podman. A separate clean Ubuntu 24.04
GitHub Actions runner executes the exact default `just` command, followed by
`make test` inside the built container.

[Container execution log](https://github.com/billalgio103-glitch/Special-Topic/actions/runs/37539035191)

**Passed on 2026-10-06.** The clean runner downloaded all sources and completed
`just`, including database construction, full verification and all three figures.
All eight container tests also passed. The run took approximately 17 minutes;
download time varies with network conditions.

The container database contained the same 1,128 images and 23,386 objects, and
its logical fingerprint exactly matched the native builds above. The SQL summary
view's floating-point averages can differ in their final digit between SQLite
builds; the stored database rows and their fingerprint matched.

See [`validation/ci-validation.json`](validation/ci-validation.json) for the
report extracted from the workflow log and
[`validation/ci-run.json`](validation/ci-run.json) for the tested source commit
and workflow result. Subsequent documentation-only commits do not change the
tested ETL code.

## ETL Part II: safetensors export

`just pipeline` now continues after the DB to `data/inputs/{train,eval}.safetensors`.
`just check` reloads both files with `safetensors.torch.load_file` and
`TensorDataset`/`DataLoader`. Local checks on the 40-image EMPS development subset
(Windows, Python 3.12, native venv, no Podman) passed:

- export and `check_tensors.py`: 32 train images (31 groups, 5 folds), 8 eval
  images (6 groups), with no group or image shared between them;
- `check_idempotency.py`: identical DB fingerprint and identical SHA-256 for both
  tensor files across two rebuilds;
- 2 HRTEM DM3 images: export to 256 x 256 kept the foreground fraction (0.60%).

The full 1,128-image tensor export has not yet been recorded here. The GitHub
Actions workflow runs it with `just pipeline` and `just check`.
