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
`TensorDataset`/`DataLoader`, re-checks them against the DB, and runs the unit tests.

### Full-source container run

[Workflow run #5](https://github.com/billalgio103-glitch/Special-Topic/actions/runs/39886805607)
on a clean Ubuntu 24.04 runner executed `just pipeline` and then `just check` for
commit `3fbec65`. **It passed on 2026-10-23** in about 17 minutes. All 13 container
tests passed.

The DB again contained 1,128 images and 23,386 objects. Its logical fingerprint,
`cab870d8...c5a3`, matches the Part I builds exactly, so the Part II changes did
not alter the database.

| File | Images | EMPS | HRTEM | Co3O4 | Groups | Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `train.safetensors` (DB train + validation) | 982 | 397 | 329 | 256 | 284 | 307 MB |
| `eval.safetensors` (DB test) | 146 | 68 | 78 | 0 | 50 | 46 MB |

No image or group appears in both files. Each train group belongs to exactly one
fold. The fold sizes are 420, 95, 144, 218 and 105 images. Fold 0 is large because
the 256 Co3O4 patches form one group, which cannot be split without leakage.

File SHA-256:

```text
train  f0c46d30c570e6b508bfb72c4251f2351d1c6f9b7878531e4149890267f88260
eval   efc1fb8d5644d5c5e70b4c05607a1e8a363aafabab097155d6dbdbfb01c4f203
```

See [`validation/ci-tensors-validation.json`](validation/ci-tensors-validation.json)
and [`validation/ci-part2-run.json`](validation/ci-part2-run.json).

An earlier attempt (run #4) was cancelled after 1 h 35 min. The NERSC HRTEM server
refused connections from the runner at the time. Run #5 used the same commit and
downloaded every asset.

### Development-subset checks

Local checks on the 40-image EMPS subset (Windows, Python 3.12, native venv, no
Podman) passed:

- export and `check_tensors.py`: 32 train images (31 groups, 5 folds), 8 eval
  images (6 groups), with no group or image shared between them;
- `check_idempotency.py`: identical DB fingerprint and identical SHA-256 for both
  tensor files across two rebuilds;
- 2 HRTEM DM3 images: export to 256 x 256 kept the foreground fraction (0.60%).

The two-build tensor comparison (`just idempotency`) has been run on this subset
only, not on the full data.
