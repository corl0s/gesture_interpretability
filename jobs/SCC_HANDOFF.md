# Handoff for the SCC Claude session

Goal: run the full-scale E0–E4 experiments (see `../README.md`) on Qwen3-VL-8B (and optionally 2B for comparison) on BU SCC GPUs.

## What to check or adapt first (written without SCC access)
1. In `scc_experiment.qsub` and `scc_setup.sh`: replace `CHANGE_ME_PROJECT` with the user's SCC project, and check the module names (`python3/3.11`, `cuda`) and the GPU flags (`gpus=1`, `gpu_c=8.0`, `gpu_memory=40G`). 8B in bf16 needs ~18 GB plus activations; any ≥24 GB GPU works.
2. Run `bash jobs/scc_setup.sh` once on a login node. It creates the venv and pre-downloads the weights into `HF_HOME`.
3. `bash jobs/submit_full_scale.sh qwen3vl-8b results_full` submits 15 jobs. E4 waits on E1-object (`-hold_jid`).
4. After the `e1*` jobs: `python experiments/e2_compare_ids.py --ids_dir results_full/qwen3vl-8b/ids` (CPU is enough).

## Expected outputs
`results_full/<model>/{e0_baseline,ids,e3_swap,e4_steer}/` holds JSON rows plus PNG plots. The E1 `.pt` files for 8B are about 1 GB per condition (float16 activations, every layer).

## Notes
- The model loader (`deixis/model.py::_load_streaming`) avoids safetensors mmap because of a Windows-only problem; it works on Linux as well.
- Rough runtime per job on one A100/L40S: E0 under 30 min, E1 under 30 min per condition, E3 a few hours (≈ 6 subsets × 37 layers forwards per pair), E4 about 1–2 h. Raise `h_rt` if E3 is slow.
