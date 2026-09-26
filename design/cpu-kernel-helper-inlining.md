# CPU Kernel Helper Inlining

**Status**: Implemented; single-host validation complete

**Issue**: [GH-1994](https://github.com/NVIDIA/warp/issues/1994)

## Motivation

Warp lowers `@wp.func` calls to C++ helpers before its embedded Clang and LLVM compiler builds CPU kernels. Some helpers accept large Warp structs by value. When LLVM keeps such a call out of line, a topology loop can repeatedly copy a dataset handle to the stack and lose optimization opportunities across the call. The cost depends on LLVM's inlining decisions, so a small change to surrounding generated code can cause a large change in execution time.

The first clear example was Warp SimData's polyhedral CGNS `count_segments` kernel. A CPU kernel specialization duplicated call sites, after which LLVM left struct-taking helpers out of line. The specialization exposed the problem, but choosing a suitable CPU inlining policy is a separate compiler concern: ordinary Warp kernels also call generated helpers, and the policy should be tested with and without that specialization.

## Requirements

- Increase steady-state CPU execution speed across representative kernels and keep regressions in cases that do not benefit to roughly 5% or less.
- Check Newton CPU simulation examples as well as Benchy data-processing operators, and report regressions for each workload separately.
- Preserve explicit `@wp.func(inline=True)` and `inline=False` behavior and low-optimization debugging behavior.
- Limit JIT compilation time and object-code growth. Measure first-call time separately from steady-state execution; first call is not compile time alone.
- Validate CUDA code generation and low-optimization CPU behavior because the prototype changes process-wide LLVM options.

## Design

### Selected policy

The branch sets two registered LLVM options in `warp/native/clang/clang.cpp`
during `initialize_llvm()`. CPU compilation and the optional LLVM-CUDA backend
call this initializer; the default NVRTC CUDA backend does not. The selected
policy is process-wide:

| LLVM option | Bundled LLVM default | Selected value |
| --- | ---: | ---: |
| `inline-threshold` | 225 | 3000 |
| `inline-cold-callsite-threshold` | 45 | 3000 |

Both thresholds matter. Raising only the generic threshold left a focused hex `iso_surface` case around 84 ms even with a value of 50,000; raising only the cold-callsite threshold to 225 brought it to about 64 ms. Raising both to 3000 brought it to about 40 ms and recovered the focused polyhedral cases. These were single-process tuning runs, not a confidence interval. LLVM's cost model still chooses which calls to inline, and the finite budgets bound code growth better than inlining every viable call.

The focused sweep below shows why the cold-callsite budget was included. Values are steady-state milliseconds from one process; the full repeated matrix later checked the selected pair:

| Generic / cold threshold | CGNS `iso_surface` | CGNS `external_faces` | Hex `iso_surface` |
| --- | ---: | ---: | ---: |
| 1000 / default | 76.5 | 702.2 | 107.5 |
| 5000 / default | 76.4 | 509.9 | 82.2 |
| 1000 / 1000 | 74.7 | 698.6 | 67.3 |
| 2000 / 2000 | 74.6 | 566.6 | 40.8 |
| 3000 / 3000 | 76.2 | 521.0 | 40.2 |

The earlier prototype is preserved at `ncapens/cpu-inlining-prototype`
(`908f5e88893caab87d76c8847f1714a6d2bb8aa6`). Its net executable-code
change relative to `ncapens/cpu-vectorization`
(`639c483a91fff54fe335435873a57f288579d763`) is the two LLVM settings.
The `ncapens/cpu-inlining-tuning` branch is based on main-derived
`76bddcae1`. Its pre-squash benchmark revision (`999d0ee6b`) applied the last
two prototype commits; their source-level `WP_FORCEINLINE` changes canceled,
leaving only the two LLVM settings and their required headers in native code.
The single-commit branch keeps those settings and adds cache-versioning,
tests, and this execution record. This main-derived pair measures the policy
without the vectorization branch's changes.

The options are set through LLVM's registered-options API inside a
`std::call_once` block. This ties the setting to LLVM option names and affects
every compilation in the process. All five bundled LLVM SDK packages in
`deps/llvm-deps.packman.xml` use version 22.1.8; the Windows x86-64 native
build and kernel runs verified that both names are registered there. The
global-setting decision and its measured CUDA and debug impact are recorded
below. CPU object filenames now include inlining-policy revision 2 in their
target hash, so objects made under the old native defaults cannot be reused.
Optional LLVM-CUDA PTX/CUBIN filenames carry the same revision; NVRTC binary
names stay unchanged. Explicit `@wp.func` inline hints remain authoritative.

### Compiler flag boundary and alternatives

Warp calls Clang `cc1` directly rather than invoking the driver. In this configuration, `-mllvm` supplied through `wp.config.cpu_compiler_flags` did not change the LLVM options: even an invalid `-mllvm` option compiled a CPU kernel successfully in the experiment. The public Clang inlining switches do not expose the two cost budgets above. `wp.config.optimization_level=3` was also slower on the focused kernels than the proposed `-O2` policy. The native LLVM-option path is therefore the working tuning mechanism in this prototype.

| Alternative | Observation on the vectorization branch |
| --- | --- |
| Force inline every eligible struct-taking CPU helper in generated source | Recovered the large losses, but changed `@wp.func` code generation by type rather than letting LLVM price individual calls. Full Benchy result: +16.4% versus main. |
| Set only `inline-threshold` | Did not recover hex `iso_surface`; the separate cold-callsite cap still applied. |
| Set `locally-hot-callsite-threshold=3000` | Slowed focused hex `iso_surface` to about 176 ms. |
| Enable `inline-all-viable-calls` | +20.9% across Benchy's 53 cases versus main, but 45.8 s summed first-call time versus 36.9 s for finite thresholds, and larger full-suite CPU objects. |
| Raise optimization level to `-O3` | Slowed the focused polyhedral and hex cases. |

### Existing measurements and their limits

The measurements below used an AMD Ryzen 7 9800X3D and Benchy's 53 applicable CPU cases. Each revision had three independent processes. Each case used one warmup and three measured repeats; the table takes the median `ms` value across processes, then the equal-weighted geometric mean of baseline/candidate speedups. Benchy and Warp SimData revisions are pinned in the reproduction section.

| Comparison | All 53 cases | `iso_surface` (6) | `external_faces` (5) | Worst case |
| --- | ---: | ---: | ---: | ---: |
| Tuned inliner versus its vectorization parent | +23.2% | +98.9% | +52.4% | -1.4% |
| Tuned inliner versus `omniverse/main` | +21.5% | +72.2% | +41.2% | -1.2% |

The parent-to-prototype comparison isolates the **native inlining settings within that branch**. It does not establish the benefit of the settings on main without vectorization. The main-to-prototype row is the total effect of both bodies of work and must not be presented as an inlining-only gain. In the parent comparison, 42 of 53 cases gained more than 5%, none lost more than 5%, and all 53 output cell and point counts matched. Counts do not prove numeric equivalence.

Across 146 Benchy CPU objects, the vectorization parent occupied 1.28 MiB, the source-level force-inline prototype 1.39 MiB, and the tuned LLVM prototype 1.22 MiB. Summed first-call times were 33.5 s for force-inline and 36.9 s for finite thresholds. In a full Warp suite run of the tuned prototype, 9463 tests ran in 739 s; the two failures were CUDA texture tests also reproduced on main. Fifty-four representative CUDA PTX outputs were byte-identical with and without the LLVM override. These observations do not replace an isolated main-branch test or broader CUDA validation.

## Reproducing and extending the benchmarks

### Warp comparison worktrees

Use the same main-derived pair for both suites. The tuned checkout is the tip
of `ncapens/cpu-inlining-tuning`; create a separate worktree for its parent.
The execution record below used the pre-squash tuned revision
`999d0ee6bdb6537fa0c55053d906432402e5c8ed`. These commands assume the
current checkout is at `C:\src\warp-cpu-inlining-tuning`:

```powershell
$warpSource = 'C:\src\warp-cpu-inlining-tuning'
$baselineRoot = 'C:\src\warp-inlining-baseline'
$baselineRevision = '76bddcae16fa5ecc7b7c32986e848a226b4322eb'
$tunedRevision = git -C $warpSource rev-parse HEAD
git -C $warpSource worktree add --detach $baselineRoot $baselineRevision
$warpRoots = @{ baseline = $baselineRoot; tuned = $warpSource }
foreach ($label in @('baseline', 'tuned')) {
    $warpRoot = $warpRoots[$label]
    if ((git -C $warpRoot rev-parse HEAD) -ne $(if ($label -eq 'baseline') { $baselineRevision } else { $tunedRevision })) {
        throw "Unexpected Warp revision: $label"
    }
    Push-Location $warpRoot
    try { uv run build_lib.py; if ($LASTEXITCODE -ne 0) { throw "Warp build failed: $label" } }
    finally { Pop-Location }
}
```

Record the full revision and native build options with each result. The
vectorization-branch pair in the experiment matrix below is a separate study;
it needs its own two worktrees and native builds.

### Benchy sources and workload

The historical Benchy measurements above used these exact revisions. The
main-derived pair built in the preceding section is the new isolated test:

| Component | Revision | Role |
| --- | --- | --- |
| [Benchy `chschultz/warp-simdata-proposed-bench`](https://gitlab-master.nvidia.com/uayachit/benchy/-/tree/chschultz/warp-simdata-proposed-bench?ref_type=heads) | `12c175888d3cbb682bd672aa07cb6cb1a9eb8873` | `examples/warp-ci/run.py` and `matrices/warp-ci.yaml` |
| [Warp SimData](https://gitlab-master.nvidia.com/omniverse/warp-simdata) | `48c90c3cd343db2776c60f0fd117f3f54decf610` | Operators and generated dataset handling |
| Warp `omniverse/main` | `4500ab5196ba6cbfa6bb68f670e6ca6efff14e08` | Total-performance baseline |
| Warp `ncapens/cpu-vectorization` | `639c483a91fff54fe335435873a57f288579d763` | Parent without inlining settings |
| Warp `ncapens/cpu-inlining-prototype` | `908f5e88893caab87d76c8847f1714a6d2bb8aa6` | Parent plus tuned settings |

These pinned Benchy sources require access to the internal repositories. As
checked on 2026-09-25, the public
[NVIDIA/warp-simdata](https://github.com/NVIDIA/warp-simdata) repository is a
0.1.0 snapshot at `dfc8400bc70d5913409142df3be4e278ccc94043`; it does
not contain the tested `48c90c3` revision, and relevant operator sources
differ. No public copy of this Benchy `warp-ci` suite was found. The Newton
examples below provide a separately reproducible public workload for issue
#1994; the historical Benchy numbers depend on these internal sources.

The `warp-ci` matrix generates six lattices in memory: hex (1.0M cells), tet (3.072M), mixed fixed-cell (0.983M), polyhedral CGNS (0.5M), polyhedral OpenFOAM (0.5M), and VTI image data (4.096M). No external dataset download is needed. It exercises nine operators: `iso_surface`, `slice`, `external_faces`, `element_centers`, `bounds`, `probe` (100,000 points), `streamlines` (1,000 seeds), `cell_to_point`, and `point_to_cell`. One combination is inapplicable, leaving 53 CPU cases. Read Benchy's `examples/warp-ci/README.md` and `matrices/warp-ci.yaml` for the current definitions. `run.py` measures Warp SimData only; Benchy's full matrix also includes CUDA and VTK.

### Benchy setup and execution on Windows

Use the two Warp worktrees built above. The commands below assume source clones
at `C:\src\benchy` and `C:\src\warp-simdata`; substitute local paths on
another machine. Create clean worktrees at the pinned revisions so the
benchmark never picks up changes from either source checkout:

```powershell
$benchySource = 'C:\src\benchy'
$benchyRoot = 'C:\src\benchy-warp-ci'
$simdataSource = 'C:\src\warp-simdata'
$simdataRoot = 'C:\src\warp-inlining-simdata'
git -C $benchySource worktree add --detach $benchyRoot 12c175888d3cbb682bd672aa07cb6cb1a9eb8873
git -C $simdataSource worktree add --detach $simdataRoot 48c90c3cd343db2776c60f0fd117f3f54decf610
```

Set up Benchy's Python dependencies and machine-local paths as described in
its `playbooks/setup.md`; that playbook uses Linux paths, so use the analogous
Windows locations here. The synthetic lattices require no dataset download.
The `simdata`, `usd`, and `vtk` extras supply the benchmark and equivalence
requirements; `cae-openusd-plugins` uses Benchy's configured internal index.
Install with `uv sync --project $benchyRoot --extra simdata --extra usd --extra vtk`
if that index is available. Run
`uv run --no-sync --project $benchyRoot benchy doctor` to confirm the required
backends and paths before collecting results. Verify that `benchy`, `warp_simdata`,
`cae_openusd_plugins`, and `pxr` import before benchmarking. The `PYTHONPATH`
below selects the local Warp checkout and pinned Warp SimData source ahead of
installed wheels. Verify their `__file__` paths too.

The pinned Benchy revision has no `uv.lock`; record the resolved dependency
versions, Python version, and internal index configuration. Run three separate
processes per Warp revision, alternating the order. Give each run a distinct
`WARP_CACHE_PATH`. This loop uses `$warpRoots`, `$benchyRoot`, and
`$simdataRoot` from the setup blocks above:

```powershell
$results = 'C:\src\warp-inlining-benchy-results'
New-Item -ItemType Directory -Force -Path $results | Out-Null
$env:UV_CACHE_DIR = Join-Path $results 'uv-cache'
Set-Location $benchyRoot
for ($run = 1; $run -le 3; $run++) {
    $order = if ($run % 2 -eq 1) { @('baseline', 'tuned') } else { @('tuned', 'baseline') }
    foreach ($label in $order) {
        $warpRoot = $warpRoots[$label]
        $env:PYTHONPATH = "$warpRoot;$benchyRoot\src;$simdataRoot"
        $env:WARP_CACHE_PATH = Join-Path $results "$label-cache-$run"
        $json = Join-Path $results "$label-$run.json"
        uv run --no-sync --project $benchyRoot python -c "import warp, warp_simdata; print(warp.__file__); print(warp_simdata.__file__)"
        if ($LASTEXITCODE -ne 0) { throw "Benchy imports failed: $label / $run" }
        uv run --no-sync --project $benchyRoot examples/warp-ci/run.py --devices cpu --json $json
        if ($LASTEXITCODE -ne 0) { throw "Benchy benchmark failed: $label / $run" }
    }
}
```

Record the CPU model, Warp revision, Python package versions, and all compiler flags with the JSON files. Avoid concurrent CPU-heavy work. The runner reports `ms` (median steady-state operator time), `first_ms`, output counts, and errors. Reject errored or missing cases; compare matching `(dataset, operation)` keys and output counts. For each case, take the median `ms` from the three processes. For an aggregate, compute `exp(mean(log(baseline_ms / candidate_ms))) - 1` across the 53 cases. Report per-operation geometric means, the worst individual regression, and the number of cases outside +/-5%; do not rely only on the aggregate. The earlier prototype produced +23.2% versus its parent and +21.5% versus main under this procedure.

For fast tuning, use `--datasets lattice-polyhedra-ci-cgns,lattice-hex-ci` and
`--operations iso_surface,external_faces`; then confirm any winner on the full
matrix. The lightweight runner checks only cell and point counts. After timing,
run the pinned full matrix with VTK as the contract reference for each revision:

```powershell
foreach ($label in @('baseline', 'tuned')) {
    $env:PYTHONPATH = "$($warpRoots[$label]);$benchyRoot\src;$simdataRoot"
    $env:WARP_CACHE_PATH = Join-Path $results "$label-equivalence-cache"
    $out = Join-Path $results "$label-equivalence"
    uv run --no-sync --project $benchyRoot benchy run warp-ci --backends warp-simdata:cpu,vtk:cpu --out $out
    if ($LASTEXITCODE -ne 0) { throw "Benchy equivalence run failed: $label" }
}
```

Inspect each run's `equivalence.json` for failed or missing cases; save these
reports beside the timing JSON files. Benchy's contract checks compare counts,
bounds, and resolved parameters with operation-specific tolerances, not full
field arrays or individual elements. Compare baseline and tuned numerical
arrays separately for representative cases; a successful process exit alone
is not the equivalence result.

### Newton CPU simulation examples

Use Newton examples from `C:\src\newton` as a second workload. Newton's tracked
`newton.examples` launcher accepts `--device cpu --benchmark SECONDS`; benchmark
mode uses a null viewer, warms up for three frames, synchronizes each timed
frame, and prints `Benchmark: ... FPS (... frames in ...s)`. It stops at the
time limit or `--num-frames`, whichever comes first. The default 100-frame cap
can make a fast example finish in a fraction of a second, so set a high frame
cap explicitly. These FPS results measure whole simulation frames and should
not be combined with Benchy's operator timings in one aggregate.

Start with `basic_urdf`, `cable_twist`, `cloth_bending`, `diffsim_drone`,
`ik_h1`, `sensor_contact`, and `softbody_hanging`. They exercise distinct
simulation paths and have produced CPU benchmark output in the local Newton
checkout. Preflight them on both Warp revisions; record unsupported examples
or asset failures rather than silently removing them from one side. Extend the
final screen to other CPU-capable Newton examples using the same fixed list on
every revision.

Use one clean, pinned Newton revision and the same assets and Python
dependencies for both Warp revisions. The local `C:\src\newton` checkout is
the source for a benchmark worktree pinned at `ab804f14b7d27775f7043ee5120520cfbf54cfab`.
Use the built Warp worktrees defined above, select each source with
`PYTHONPATH`, and verify the import paths. Each example invocation is a
separate process. Alternate the
revision order across three runs and give each revision and run a distinct Warp
kernel cache:

```powershell
$newtonSource = 'C:\src\newton'
$newtonRoot = 'C:\src\newton-inlining-bench'
$newtonRevision = 'ab804f14b7d27775f7043ee5120520cfbf54cfab'
$results = 'C:\src\warp-inlining-newton-results'
$examples = @(
    'basic_urdf', 'cable_twist', 'cloth_bending', 'diffsim_drone',
    'ik_h1', 'sensor_contact', 'softbody_hanging'
)
git -C $newtonSource worktree add --detach $newtonRoot $newtonRevision
New-Item -ItemType Directory -Force -Path $results | Out-Null
$env:UV_CACHE_DIR = Join-Path $results 'uv-cache'
git -C $newtonRoot rev-parse HEAD  # record this revision with the results
Set-Location $newtonRoot
uv sync --locked --extra examples

for ($run = 1; $run -le 3; $run++) {
    $order = if ($run % 2 -eq 1) { @('baseline', 'tuned') } else { @('tuned', 'baseline') }
    foreach ($label in $order) {
        $env:PYTHONPATH = $warpRoots[$label]
        $env:WARP_CACHE_PATH = Join-Path $results "$label-cache-$run"
        uv run --no-sync python -c "import warp, newton; print(warp.__file__); print(newton.__file__)"
        foreach ($example in $examples) {
            $log = Join-Path $results "$label-$run-$example.log"
            uv run --no-sync -m newton.examples $example --device cpu --benchmark 10 --num-frames 100000 *> $log
            if ($LASTEXITCODE -ne 0 -or -not (Select-String -LiteralPath $log -Pattern '^Benchmark: ' -Quiet)) {
                throw "Newton benchmark failed: $label / $run / $example"
            }
        }
    }
}
```

Keep the logs with the CPU model, Newton and Warp revisions, dependency
versions, compiler flags, and asset revisions. Extract FPS, timed frames, and
elapsed time from each `Benchmark:` line. For each example, compare the median
FPS across the three runs and report
`(tuned_fps / baseline_fps - 1) * 100`, the worst regression, and run-to-run
spread. A separate geometric mean of Newton FPS ratios can summarize the
screen, but it must not hide individual losses over roughly 5%. Flag runs
with few timed frames or much less than ten seconds of timed work; increase the
duration for those examples and rerun both revisions. Avoid concurrent CPU-heavy
work. FPS does not establish numerical correctness. From the pinned Newton
worktree, run these existing example tests with each Warp revision selected in
`PYTHONPATH` and a separate cache, then investigate any new failure:

```powershell
uv run --no-sync -m unittest newton.tests.test_examples -k basic_urdf -k cable_twist -k cloth_bending -k diffsim_drone -k ik_h1 -k sensor_contact -k softbody_hanging
```

### Comparison matrix

The acceptance comparison used a two-by-two matrix on the same host and dependency revisions:

| Base compiler | Default inlining | Tuned inlining |
| --- | --- | --- |
| Main, without CPU vectorization changes | `76bddcae1` | `999d0ee6b` (pre-squash benchmark revision) |
| Vectorization branch | `639c483a9` | `908f5e888` |

The top-row comparison answers whether the policy helps CPU kernels independently; the bottom row measures interaction with the vectorization specialization. Both Benchy and Newton used the same host and pinned dependencies with alternating revision order. The preliminary sweep compared finite generic and cold thresholds with the default, source-level force-inline, and `inline-all-viable-calls`; representative native objects were disassembled. The roughly 5% regression guard applies to individual cases in both workloads, not just their separate aggregate results.

### Acceptance criteria

1. Publish all per-case Benchy timings and all per-example Newton FPS values for the isolated top-row comparison, with three process runs, spread, machine details, revisions, and package versions. Explain any case slower by more than roughly 5%; retune or reject a policy with material unexplained regressions. Repeat the winning threshold pair on the full matrices and on a second CPU if available.
2. Measure compilation separately from runtime. Use fresh per-revision caches and Warp's debug-level `Compile x86` and module-load timers for representative kernels in three processes; collect the resulting CPU object sizes from those caches. Report per-kernel and total compile time and object bytes against baseline, alongside Benchy's `first_ms`. Some kernels compile during setup, and `first_ms` includes other work, so it cannot stand in for JIT cost. Set an acceptable compile-time and size budget before choosing the final thresholds.
3. Run Benchy's contract checks, independent numeric array comparisons, and the corresponding Newton example tests under both revisions. Run `uv run --extra dev -m warp.tests -s autodetect` for the candidate and baseline; use a machine with CUDA for CUDA coverage and compare representative PTX from both NVRTC and LLVM-CUDA with fresh caches. Investigate any differing output or test failure. Run `uv run -m unittest warp.tests.test_func_inline` for both revisions, and add focused CPU native-object checks showing `inline=True` and `inline=False` remain effective at release `-O2` and debug `-O0`. Keep baseline failures separate from new failures.
4. Decide whether the global LLVM options are acceptable for CUDA and low-optimization CPU builds. If they are not, configure the inliner only in the CPU optimization pipeline. For the released policy, change the CPU object-cache identity (for example, the `warp-cpu-target-v1` token in `_get_cpu_target_hash()` in `warp/_src/context.py`) when the native defaults change, and verify that old cached objects are not reused. Record how `cpu_compiler_flags` interacts with the setting. Add a `1994.changed.md` Towncrier fragment for the user-visible performance change.

The execution record below applies these gates to the main-derived pair and
the vectorization pair. The earlier prototype measurements remain for context;
the isolated main-derived results determine the policy decision.

## Isolated main-derived execution record (2026-09-26)

The baseline was `76bddcae16fa5ecc7b7c32986e848a226b4322eb`; the tuned checkout was based on the pre-squash revision `999d0ee6bdb6537fa0c55053d906432402e5c8ed`. The final branch retains the same numeric LLVM thresholds; its later cache identity, tests, documentation, and explicit threshold assignments were not part of these timings. Benchy was pinned at `12c175888d3cbb682bd672aa07cb6cb1a9eb8873`, Warp SimData at `48c90c3cd343db2776c60f0fd117f3f54decf610`, and Newton at `ab804f14b7d27775f7043ee5120520cfbf54cfab`. The host was an AMD Ryzen 7 9800X3D with an NVIDIA GeForce RTX 5080, Windows, Python 3.12.12, LLVM 22.1.8, CUDA Toolkit 13.0, and Warp 1.19.0.dev0. Performance runs used Warp's default release mode, CPU `-O2`, and effective `-march=native`, without additional CPU compiler flags. The Benchy environment used NumPy 2.5.3, DuckDB 1.5.5, h5py 3.16.0, PyYAML 6.0.3, and VTK 9.7.0; the locked Newton environment used NumPy 2.5.0 and SciPy 1.17.1. Newton imported the Warp source checkout through `PYTHONPATH`, overriding its installed Warp wheel. Runs used separate fresh kernel caches and alternated revision order.

The full Benchy `simdata`, `usd`, and `vtk` extras could not be installed because NVIDIA Artifactory returned HTTP 403 for `cae-openusd-plugins`. A workspace-local environment with the pinned Benchy and Warp SimData source checkouts plus public dependencies ran the synthetic CPU and VTK cases; functionality requiring that private plugin was outside this validation.

### Benchy CPU steady-state timing

Across 53 matching CPU cases, the equal-weighted geometric mean of median speedups was **+14.98%**. Thirty-nine cases improved by more than 5%, and none regressed by more than 5%. The worst median change was OpenFOAM polyhedral `point_to_cell` at **-2.24%**; its first tuned run was slower, but the three-run medians differed by 0.124 ms. Output cell and point counts matched for every case. Each cell below lists the three process-level `ms` values; each process used Benchy's warmup and three measured repeats.

| Dataset | Operation | Baseline ms, runs 1/2/3 | Tuned ms, runs 1/2/3 | Median gain |
| --- | --- | ---: | ---: | ---: |
| lattice-hex-ci | bounds | 3.686 / 3.685 / 3.701 | 3.688 / 3.680 / 3.723 | -0.1% |
| lattice-hex-ci | cell_to_point | 14.606 / 18.132 / 14.428 | 13.668 / 18.097 / 13.878 | +5.3% |
| lattice-hex-ci | element_centers | 11.893 / 11.604 / 11.302 | 9.763 / 9.959 / 10.169 | +16.5% |
| lattice-hex-ci | external_faces | 600.192 / 582.508 / 571.111 | 394.640 / 400.059 / 401.904 | +45.6% |
| lattice-hex-ci | iso_surface | 104.344 / 106.543 / 108.600 | 47.007 / 51.892 / 47.087 | +126.3% |
| lattice-hex-ci | point_to_cell | 9.107 / 9.252 / 10.100 | 8.531 / 8.585 / 8.407 | +8.4% |
| lattice-hex-ci | probe | 87.252 / 74.984 / 90.109 | 84.337 / 79.501 / 70.861 | +9.7% |
| lattice-hex-ci | slice | 38.559 / 38.313 / 39.460 | 35.388 / 35.004 / 35.205 | +9.5% |
| lattice-hex-ci | streamlines | 165.984 / 168.478 / 172.937 | 131.095 / 129.730 / 130.698 | +28.9% |
| lattice-mixed-ci | bounds | 1.274 / 1.275 / 1.272 | 1.277 / 1.307 / 1.272 | -0.3% |
| lattice-mixed-ci | cell_to_point | 9.370 / 9.431 / 9.380 | 8.547 / 8.550 / 8.754 | +9.7% |
| lattice-mixed-ci | element_centers | 8.641 / 8.475 / 8.565 | 8.691 / 7.822 / 7.567 | +9.5% |
| lattice-mixed-ci | external_faces | 523.477 / 544.081 / 551.699 | 362.211 / 406.400 / 359.229 | +50.2% |
| lattice-mixed-ci | iso_surface | 72.417 / 83.116 / 74.870 | 39.012 / 40.241 / 41.799 | +86.1% |
| lattice-mixed-ci | point_to_cell | 6.805 / 6.891 / 6.868 | 6.212 / 6.227 / 6.016 | +10.6% |
| lattice-mixed-ci | probe | 92.649 / 99.265 / 88.705 | 76.730 / 81.992 / 106.877 | +13.0% |
| lattice-mixed-ci | slice | 28.051 / 32.165 / 29.221 | 27.645 / 28.284 / 25.403 | +5.7% |
| lattice-mixed-ci | streamlines | 302.171 / 311.838 / 308.844 | 231.489 / 237.450 / 236.135 | +30.8% |
| lattice-polyhedra-ci-cgns | bounds | 3.694 / 3.876 / 3.694 | 3.669 / 3.680 / 3.666 | +0.7% |
| lattice-polyhedra-ci-cgns | cell_to_point | 13.050 / 13.258 / 13.524 | 12.561 / 12.161 / 12.638 | +5.6% |
| lattice-polyhedra-ci-cgns | element_centers | 8.179 / 9.341 / 8.510 | 7.664 / 7.739 / 7.654 | +11.0% |
| lattice-polyhedra-ci-cgns | external_faces | 664.287 / 689.095 / 732.226 | 553.304 / 531.624 / 559.471 | +24.5% |
| lattice-polyhedra-ci-cgns | iso_surface | 92.157 / 87.685 / 85.551 | 82.621 / 83.659 / 81.901 | +6.1% |
| lattice-polyhedra-ci-cgns | point_to_cell | 7.721 / 7.566 / 7.714 | 6.908 / 6.836 / 6.842 | +12.8% |
| lattice-polyhedra-ci-cgns | probe | 195.894 / 202.452 / 224.900 | 170.681 / 197.568 / 185.553 | +9.1% |
| lattice-polyhedra-ci-cgns | slice | 42.038 / 43.703 / 54.339 | 37.651 / 37.496 / 37.827 | +16.1% |
| lattice-polyhedra-ci-cgns | streamlines | 462.179 / 466.167 / 479.673 | 437.076 / 420.476 / 426.498 | +9.3% |
| lattice-polyhedra-ci-foam | bounds | 3.700 / 3.692 / 3.718 | 3.694 / 3.691 / 3.690 | +0.2% |
| lattice-polyhedra-ci-foam | cell_to_point | 10.743 / 10.588 / 11.097 | 10.984 / 10.740 / 10.627 | +0.0% |
| lattice-polyhedra-ci-foam | element_centers | 7.317 / 7.271 / 7.260 | 6.313 / 6.432 / 6.483 | +13.0% |
| lattice-polyhedra-ci-foam | external_faces | 568.033 / 573.792 / 596.551 | 519.184 / 515.787 / 553.698 | +10.5% |
| lattice-polyhedra-ci-foam | iso_surface | 83.996 / 81.217 / 97.638 | 75.394 / 74.776 / 74.775 | +12.3% |
| lattice-polyhedra-ci-foam | point_to_cell | 5.420 / 5.379 / 5.612 | 6.632 / 5.423 / 5.545 | -2.2% |
| lattice-polyhedra-ci-foam | probe | 208.584 / 220.948 / 192.669 | 206.197 / 199.007 / 189.285 | +4.8% |
| lattice-polyhedra-ci-foam | slice | 37.598 / 37.246 / 43.308 | 37.652 / 34.806 / 35.184 | +6.9% |
| lattice-polyhedra-ci-foam | streamlines | 374.869 / 375.571 / 378.151 | 377.788 / 372.204 / 376.435 | -0.2% |
| lattice-tet-ci | bounds | 1.938 / 1.933 / 2.036 | 1.950 / 1.951 / 1.978 | -0.6% |
| lattice-tet-ci | cell_to_point | 25.124 / 24.786 / 24.634 | 22.852 / 21.868 / 22.013 | +12.6% |
| lattice-tet-ci | element_centers | 23.357 / 23.684 / 24.379 | 20.975 / 21.215 / 31.708 | +11.6% |
| lattice-tet-ci | external_faces | 1107.070 / 1126.211 / 1116.958 | 859.652 / 831.514 / 863.403 | +29.9% |
| lattice-tet-ci | iso_surface | 152.380 / 144.779 / 149.742 | 90.086 / 91.467 / 90.097 | +66.2% |
| lattice-tet-ci | point_to_cell | 19.140 / 19.025 / 22.076 | 16.331 / 16.402 / 16.345 | +17.1% |
| lattice-tet-ci | probe | 119.309 / 124.218 / 110.937 | 116.902 / 106.153 / 114.947 | +3.8% |
| lattice-tet-ci | slice | 71.228 / 71.204 / 71.284 | 65.650 / 69.659 / 71.749 | +2.3% |
| lattice-tet-ci | streamlines | 137.293 / 135.146 / 145.822 | 93.693 / 80.444 / 89.012 | +54.2% |
| lattice-vti-ci | bounds | 18.295 / 18.034 / 18.219 | 16.340 / 16.223 / 16.207 | +12.3% |
| lattice-vti-ci | cell_to_point | 67.710 / 67.473 / 67.718 | 60.628 / 62.171 / 60.134 | +11.7% |
| lattice-vti-ci | element_centers | 113.760 / 114.193 / 112.876 | 117.583 / 114.849 / 114.462 | -0.9% |
| lattice-vti-ci | iso_surface | 199.557 / 202.589 / 197.471 | 161.665 / 146.964 / 147.827 | +35.0% |
| lattice-vti-ci | point_to_cell | 34.070 / 34.327 / 34.002 | 31.810 / 33.170 / 31.454 | +7.1% |
| lattice-vti-ci | probe | 5.041 / 5.030 / 5.064 | 5.011 / 5.412 / 5.074 | -0.7% |
| lattice-vti-ci | slice | 220.471 / 221.136 / 221.554 | 224.715 / 213.094 / 213.653 | +3.5% |
| lattice-vti-ci | streamlines | 74.044 / 73.972 / 75.360 | 64.104 / 63.466 / 62.554 | +16.7% |

Summed Benchy `first_ms` medians were 29.951 s baseline and 32.660 s tuned (**+9.0%**); this includes work outside native compilation. Across 144 cached CPU objects from the first process, total bytes grew from 894,288 to 1,088,768 (**+21.7%**). A separate nine-case, three-process profile measured 47 native CPU builds per process: median summed native compile time rose from 6.666 to 7.188 s (**+7.8%**), and their object bytes grew from 221,984 to 245,872 (**+10.8%**). All 141 paired generated C++ files in this profile were byte-identical, so the compiler policy explains the native build difference. The representative polyhedral `count_segments` module took 166.4 to 186.0 ms to compile (**+11.7%**) and grew from 5,264 to 7,336 bytes (**+39.4%**). Its generated loop passes a large dataset handle by value to `segment_from_entry`, but disassembly shows that helper was already inlined in both revisions. In this module, the new policy inlined the 2,141-byte inner kernel into its exported wrapper; that duplication accounts for nearly all 2,072 bytes of object growth. The largest absolute object increase among the profiled modules was 3,976 bytes.

The `external_faces` `classify_face_flags` object shows the motivating case directly: its baseline worker calls two out-of-line helpers that take `DatasetHandle` by value, `ElementAPI__get_face` and `load_face_nodes`. Both helper symbols and their call relocations disappear in the tuned worker. The worker grows as their bodies are inlined, while the complete object shrinks from 7,448 to 7,032 bytes. A hex `iso_surface` worker similarly loses an out-of-line `get_edge_key` helper that takes `DatasetHandle` by value; five baseline `memcpy` relocations also disappear. The affected call sites therefore differ by kernel; the policy does more than one kind of inlining.

Selected per-module medians from the three independent native-build profiles show where the extra work is concentrated:

| CPU module | Build time, baseline to tuned | Object bytes, baseline to tuned |
| --- | ---: | ---: |
| Polyhedral `count_triangles` | 277.6 to 301.7 ms (+8.7%) | 13,760 to 17,736 (+28.9%) |
| `iso_surface` `emit_intersections` (variant 1) | 148.7 to 179.0 ms (+20.4%) | 5,992 to 9,208 (+53.7%) |
| `element_faces` `make_face_keys` | 186.4 to 225.6 ms (+21.1%) | 6,120 to 9,192 (+50.2%) |
| `iso_surface` `emit_intersections` (variant 2) | 153.8 to 196.9 ms (+28.0%) | 7,184 to 9,992 (+39.1%) |
| Polyhedral `count_segments` | 166.4 to 186.0 ms (+11.7%) | 5,264 to 7,336 (+39.4%) |
| `data_models.sids.utils` | 1,069.8 to 1,121.0 ms (+4.8%) | 9,824 to 10,224 (+4.1%) |

For final selection on this host, accept at most 20% additional aggregate native compilation time on the nine representative cases and 25% additional aggregate CPU object bytes across the full 53-case Benchy matrix, provided the steady-state and correctness gates also pass. The measured +7.8% and +21.7% stay within those budgets. Individual small modules can grow more in percentage terms and are assessed by their absolute increase.

### Newton CPU simulation timing

All 42 ten-second runs completed without an early frame cap. The equal-weighted geometric mean of seven per-example median FPS ratios was **+7.69%**. No median regression exceeded 5%; `softbody_hanging` was lowest at **-3.43%**, and its baseline and tuned run ranges overlapped. Each cell lists FPS from three independent processes.

| Newton example | Baseline FPS, runs 1/2/3 | Tuned FPS, runs 1/2/3 | Median gain |
| --- | ---: | ---: | ---: |
| basic_urdf | 102.20 / 102.60 / 102.50 | 117.40 / 116.70 / 117.70 | +14.54% |
| cable_twist | 63.50 / 63.30 / 63.60 | 71.30 / 71.70 / 72.20 | +12.91% |
| cloth_bending | 375.10 / 376.50 / 370.90 | 395.90 / 397.10 / 398.30 | +5.87% |
| diffsim_drone | 84.30 / 84.50 / 84.10 | 102.00 / 102.60 / 102.60 | +21.71% |
| ik_h1 | 493.10 / 497.30 / 495.40 | 516.50 / 517.30 / 517.00 | +4.36% |
| sensor_contact | 323.30 / 326.50 / 328.50 | 325.90 / 328.70 / 326.60 | +0.03% |
| softbody_hanging | 4.95 / 4.67 / 5.00 | 4.73 / 4.78 / 5.02 | -3.43% |

### Vectorization branch interaction

The same pinned Benchy and Warp SimData checkouts and package environment were used for three new alternating CPU runs of `639c483a91fff54fe335435873a57f288579d763` (vectorization parent) and `908f5e88893caab87d76c8847f1714a6d2bb8aa6` (inlining prototype). Both native libraries were rebuilt with the same `build_lib.py --quick` settings before timing. Across 53 matching cases, the finite inlining policy improved the geometric mean by **+23.09%** on this branch, close to the earlier +23.2% observation. Thirty-nine cases improved by more than 5%, none regressed by more than 5%, and all output counts matched; worst was OpenFOAM polyhedral `point_to_cell` at **-1.56%**. The policy's ratio of speedup factors was 7.05% stronger than on the main-derived pair, showing a positive interaction with the vectorization changes. Summed `first_ms` medians rose from 35.341 to 37.257 s (**+5.42%**), while the 144 cached CPU objects shrank from 1,333,904 to 1,271,632 bytes (**-4.67%**).

The pinned Newton examples were also run on both vectorization revisions, with three alternating process runs per revision and fresh caches. All 42 timed sections lasted 10.00–10.08 s, with no early stop or frame cap. The geometric mean of seven median-FPS ratios improved **+8.49%**, 0.80 percentage points more than on the main-derived pair. No median regression exceeded 5%; the worst, `softbody_hanging`, was **-0.90%**. Each cell lists FPS from three independent processes.

| Newton example | Vectorization parent FPS, runs 1/2/3 | Inlining prototype FPS, runs 1/2/3 | Median gain |
| --- | ---: | ---: | ---: |
| basic_urdf | 104.10 / 104.50 / 103.30 | 117.30 / 118.50 / 118.50 | +13.83% |
| cable_twist | 81.50 / 79.10 / 79.80 | 97.70 / 98.30 / 98.50 | +23.18% |
| cloth_bending | 380.30 / 384.00 / 386.80 | 402.80 / 404.50 / 403.60 | +5.10% |
| diffsim_drone | 85.40 / 85.60 / 85.60 | 97.80 / 98.20 / 98.60 | +14.72% |
| ik_h1 | 493.10 / 496.20 / 500.10 | 515.70 / 521.50 / 518.60 | +4.51% |
| sensor_contact | 322.30 / 326.70 / 329.40 | 325.80 / 330.10 / 330.80 | +1.04% |
| softbody_hanging | 11.00 / 11.10 / 11.10 | 11.00 / 11.00 / 11.00 | -0.90% |

### Correctness validation

- Benchy's full synthetic CPU/VTK contract completed 12/12 jobs with zero errors under each revision. Each had 53 operator cases per backend plus six load cases, giving 59 case IDs and 118 backend verdicts; every verdict was comparable under Benchy's count, bounds, and parameter checks. Direct baseline-to-tuned comparison found no count, bounds, or parameter difference in the 53 Warp SimData cases. VTK OpenFOAM `streamlines` point counts differed by 0.13% across the two independent runs while its cells and parameters matched; VTK's point count was outside that case's contract. These checks do not establish equality of every output value.
- All 15 float32 output arrays from 12 hex, CGNS polyhedral, and OpenFOAM polyhedral cases passed full-element comparison with `rtol=1e-5` and `atol=1e-6`, including all 500,000 OpenFOAM `point_to_cell` values.
- For hex and CGNS polyhedral `iso_surface` and `external_faces`, all 24 captured output arrays were bitwise identical across revisions, including point coordinates, face winding/connectivity, source element and node IDs, and iso field values. The CGNS iso operator reported the same nine skipped invalid polyhedra under each revision. Streamline arrays remain outside the direct numeric comparison.
- Both revisions passed all 16 selected Newton example tests on CPU and CUDA with no skips or failures.
- A representative struct-taking CUDA helper produced byte-identical PTX under baseline and tuned Warp with both NVRTC and optional LLVM-CUDA compilation.
- A force-inlined helper in a CPU debug kernel compiled, launched, and produced the expected 16 values under both revisions using fresh caches.
- Focused Warp inline tests passed 10/10 on baseline and 12/12 on tuned; the tuned run includes two new CPU native-object tests. A final focused retest passed 12/12 after those tests were extended to inspect both `inline=True` and `inline=False` at CPU release `-O2` and debug `-O0`. The full Warp suites ran 9,455 and 9,458 tests respectively. Each had the same two CUDA texture array failures at optimization level 3 and 74 skips, with no new tuned failure.

The results above describe one CPU and one GPU; a second CPU was unavailable.

### Implementation decision

Keep the finite `inline-threshold=3000` and `inline-cold-callsite-threshold=3000` settings in `initialize_llvm()`. The isolated main-derived pair improved Benchy by 14.98% and Newton by 7.69%; the vectorization pair improved them by 23.09% and 8.49%. Neither pair had a case more than 5% slower in either workload. The +7.8% representative native-build cost and +21.7% full-matrix CPU object size stay below the stated 20% and 25% limits. This supports the process-wide LLVM setting on the tested host.

The default NVRTC CUDA path does not call `initialize_llvm()`. A representative struct-taking CUDA helper produced identical PTX with both NVRTC and LLVM-CUDA across baseline and tuned revisions; the full Warp suite found no new CUDA failure. The debug CPU run produced identical values, and native-object tests show explicit inline hints remain effective at both optimization levels. LLVM-CUDA binaries receive a policy-specific cache name because that optional backend does call the initializer. External MathDx LTO artifacts are compiled through separate native entry points and do not depend on these LLVM options.

The native setting is a default for the process, not a per-module option. Warp passes `cpu_compiler_flags` to Clang `cc1`; the tested `-mllvm` flags did not override these LLVM inlining thresholds, while `-march=native` still selects the host target and participates in CPU cache identity. Debug mode still selects `-O0`, where the heuristic inliner is not run. The CPU target-hash revision was raised from 1 to 2 so a cache shared with the old native defaults rebuilds CPU objects; optional LLVM-CUDA PTX and CUBIN names are similarly revised.

Only Windows x86-64 with bundled LLVM 22.1.8 was built and timed here. The five bundled SDK packages pin the same LLVM version, but other operating systems, architectures, and custom LLVM builds need their normal CI coverage. The private Benchy OpenUSD plugin was unavailable, so this record covers its synthetic CPU/VTK matrix and direct output checks, not plugin-dependent workflows.
