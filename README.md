# ridge

ridge predicts how fast a tensor-core GEMM will run on an A100 before compilation. Using a tile configuration, it returns sustained throughput along with what hardware resource limits it: tensor-core issue, shared memory bandwidth, HBM, occupancy, or an underfilled grid.

This project was mostly just to learn how profilers are made and it doesn't really constitute a production ready product. It has a somewhat respectable mean error of 16%, which is still a bit worse than other profilers. It also is able to use the data to losslessly prune the autotuning search, reducing the number of searches by about 2.4x.

## Architecture

```
4 microbenchmarks -> hardware constants
tile config -> cost model -> throughput + bottleneck -> rank -> benchmark top k -> best config
```

## The four pieces

### Model

The model works out how long one K-step takes for a single block, then scales that up to the whole problem. Tensor time comes from the measured cost of one `mma` instruction. Shared memory time comes from how many bytes `ldmatrix` has to pull out.

Those two get added together rather than overlapped, which is the one thing here I would call a real finding. Copies from global memory run in the background, so those do hide behind compute, but `ldmatrix` does not, because the very next `mma` needs the registers it just wrote. Assuming they overlap gives 36% mean error. Adding them gives 30%, with nothing else changed.

After that, four things scale the number down: too few warps to hide latency, a grid that doesn't divide evenly across the 108 SMs, time spent filling and draining the pipeline, and how much of the data gets reused out of L2 instead of refetched. The answer is whichever is smaller, the compute limit or the bandwidth limit, and whichever of the four penalties is largest gets reported as the bottleneck.

### Calibrate

Four small CUDA programs measure the constants on the GPU you are running (some A100 variant), instead of directly using the spec sheet as I found in my testing that there was some noticeable delta from the spreadsheet (~5%). The spec numbers are only kept around as a sanity range so obviously broken benchmarks can be caught.

| Constant | Measured | Datasheet | |
|---|---|---|---|
| `mmaCyclesPerInst` | 2.026 cycles | implies 307.9 vs 312 TFLOP/s | 98.7% |
| `smemBytesPerCycle` | 127.88 B/cycle/SM | 128 theoretical (32 banks x 4 B) | 99.9% |
| `hbmBytesPerSec` | 1.462 TB/s | 1.555 TB/s | 94.0% |
| `warpsNeededToHide` | 4 warps/SM | saturation point, swept across ILP 1-8 | |

Initially, I had some trouble with the MMA one. For example, if the loop keeps accumulating into the same registers then each instruction waits on the previous one, so you end up timing how long an instruction takes to finish rather than how fast they can be issued, and the constant comes out way larger than it should be. The solution I came up with is to keep eight independent warp accumulators going.

### Autotune

The model ranks all the candidate configs, then only the top few get benchmarked, and the fastest of those wins.

The reason for splitting it up like this is that the model has some gaps, so it not yet good enough to just pick one by itself. I suspect that some of this is due to the ties, as the swizzle isn't modelled so some of the configs have the same predicted performance. There are some other sources of this, but this is likely one of the issues, and this could be solved by adding an input that can break those ties, but I haven't got around to doing that yet.

### Validate

To check any of this I made a simple FP16 tensor core GEMM. It uses a multi-stage `cp.async` pipeline, `ldmatrix` to get operands into registers, `mma.sync.m16n8k16` with FP32 accumulate, 16-byte row padding to avoid bank conflicts, and a block swizzle to get better L2 hit rates. It manages to get to 73% of cuBLAS, although in the future I want to try to at least push for ~85%, although I haven't quite figured out (definitively) what needs to change to get that.

## Results

### Autotuning

 Regret means how much slower the picked config is than the best one that was actually measured for that shape.

| top k | benchmarks | vs exhaustive | mean regret | worst regret | exact optimum |
|---|---|---|---|---|---|
| 1 | 14 | 24.0x | 8.66% | 33.95% | 1/14 |
| 3 | 42 | 8.0x | 3.19% | 13.89% | 8/14 |
| 5 | 70 | 4.8x | 3.08% | 13.89% | 8/14 |
| **10** | **140** | **2.4x** | **0.00%** | **0.05%** | **13/14** |
| 20 | 280 | 1.2x | 0.00% | 0.00% | 14/14 |

At k=10 you get the same config a full search would have found on 13 of the 14 shapes, and on the last one you lose 0.05%.

### Kernel

| | |
|---|---|
| 4096 cubed | 175.0 TFLOP/s, **72.9% of cuBLAS** |
| best measured | 181.0 TFLOP/s |
| K=128 | **106.2% of cuBLAS** |
| register spills | none, all 24 variants |

Somewhat surprisingly, for K=128 the kernel I made for testing is slightly better than cuBLAS, but I also guess in some ways that's expected since cuBLAS is meant for larger K-values, so realistically, if there is one I perform better on it should be for small K.

## How it compares

### Against a full search, and against rules of thumb

| Comparison | Result | Why it matters |
|---|---|---|
| Exhaustive autotuning | Same config on 13/14 shapes, 2.4x fewer benchmarks | Despite not being 100% accurate, the best config is still selected. |
| Occupancy and roofline heuristics | 9.1% and 10.9% regret at k=8, against 3.6% for picking at random | Both prefer large tiles, but large tiles also mean fewer blocks, which means GPU idling, which might cause random picking winning sometimes |
| Nsight Compute bottleneck labels | 8/8 on `SMEM` and `WAVES` | Measured warp counts also matched what the model expected on all 12 profiled configs. |

### Against published models

I used some papers to try to understand what profilers are doing these days (or where they are going). I didn't understand a lot of the math, but I was still able to use some of their findings here. TileSight is one paper I referred to a lot, it's not out yet, but it should have a repo once the paper gets published

| Comparison | Result | Details | How to close it |
|---|---|---|---|
| TileSight (single-GPU GEMM MAPE) | 12.35% vs 16.38% | About 4% behind the accuracy of TileSight | Not entirely sure, I suspect it has to do with dependencies b/w operations |
| tritonBLAS (analytical selection) | 94.7% of exhaustive vs 91.3% | 5% slower on tritonBLAS vs 9% slower for my model. This is if we just guess a config | Not entirely sure, it could be due to how the model is biasing towards configs that are some how overutilizing resources, and the lack of tie breakers for some configs means we can't differentiate them. Need to investigate further. |

## Benchmarks

```bash
./setup-box.sh

# Kernel correctness against a float64 reference, then throughput vs cuBLAS
./build/check-gemm

# measurement sweep, 336 configs, takes around 4 minutes
./build/measure

# Model accuracy and per-shape ranking against the sweep
python3 bench/validate.py

# Autotuner: shortlist size vs regret.
python3 tools/ridge-autotune.py

# Bottleneck labels vs Nsight Compute, 12 pre-registered configs
bash bench/run-ncu-validation.sh

# ridge as a pruner for Triton's GEMM autotuner
# this is just a test for fun, I never really used this nor did I intend for this to be done
python3 bench/triton-compare.py
```

## Getting started

Build is already setup for Docker (or you can also do it without Docker, if you want)

```bash
docker compose build                                    # one-time
docker compose run --rm dev bash -c "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"

docker compose run --rm dev ./build/ridge-predict \
    --gpu a100 --m 4096 --n 4096 --k 4096 \
    --bm 128 --bn 128 --bk 32 --warpm 64 --warpn 64 --stages 3
```

```
predicted:       120.2 TFLOP/s   bottleneck: SMEM
peak tensor:     311.9   compute ceiling:    120.2   HBM ceiling:   2304.0
smem eff: 0.67   occ factor: 0.75   arith intensity: 1152.0 FLOP/byte
grid: 1024 CTAs in 324 slots   waves: 3.16   wave eff: 0.790
step: 768 cyc = 512 mma + 256 ldmatrix, in series
envelope eff: 0.975   128 K-steps   prologue 832 cyc   epilogue 1663 cyc
```

Pass `--hw data/hardware/a100-sxm4-40gb.json` to use measured constants instead of datasheet placeholders.

## Testing

```bash
docker compose run --rm dev ./build/test-model
```

`test-model` runs the model on one config and checks each intermediate value against numbers I worked out by hand (or I guess by calculator).

`check-gemm` tests the kernel against a float64 reference. It starts with a self test, feeding the comparison a few deliberately wrong results to make sure they get rejected, this is mostly just a sanity check.

## Scope

One GPU (A100), one data type (FP16). The model was written around the kernel in `bench/kernels/`, so a kernel that overlaps its work differently is outside what this covers. There is also no term for the block swizzle, so it predicts the same number for every swizzle setting even though they measure differently, this is something I hope to address in the future

## Next steps

These are ranked in priority (Number 1 being highest priority or what I believe might be the most impactful).

**1. Work out what happens at 4 active warps.** This would help close the error and it's the only mistake in this design that is major enough to make a significant impact. 42 of 336 configs over predict by 39%, at exactly 4 active warps per SM, and without them the model sits at 13.16%.

**2. Model the block swizzle.** There is no field for it in `GemmConfig`, so all six swizzle settings in the sweep get the same prediction even though they measure differently. The L2 reuse term already assumes the resident blocks cover a roughly square region, and the swizzle is exactly what controls that, so this is more about making an existing assumption explicit than adding anything new. This should be a relatively small amount of work, so realistically this might be number 1 in priority.

**3. Track dependencies between operations.** Right now the model hard codes one dependency, `ldmatrix` before `mma`, and assumes everything else overlaps perfectly. I tried breaking the work down per operation to see if that was the gap and it didn't change anything. But this requires a pretty major rewrite, but it could also be a major benefit to the project as it solved a lot of the problems it currently has (but I put it as number 3 since it is such a large change)
