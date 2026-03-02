# Prime Steering Quick Guide

This guide runs activation steering on a Prime GPU VM with:
- vLLM-hosted HarmBench judge (`8002`)
- local target model execution in `scripts/steer_results.py`

## 1) Setup on Prime

```bash
prime availability list --gpu-type H100_80GB
prime pods create --id <ID> --disk-size 500 --name run-steering
prime pods status <POD_ID>
ssh <SSH_STRING>

git clone https://<USERNAME>:<TOKEN>@github.com/05rentao/prompt-optimization.git
cd prompt-optimization
git checkout baselines-edit

uv sync
mkdir -p logs outputs vectors data
```

If any model is gated:

```bash
export HF_TOKEN=hf_xxx
```

## 2) Run Steering

Default run:

```bash
chmod +x scripts/launch_steering.sh
./scripts/launch_steering.sh
```

Custom CLI run (passed through to `scripts/steer_results.py`):

```bash
./scripts/launch_steering.sh \
  --target-layers 14 15 16 \
  --steering-coefficient 1.5 \
  --max-new-tokens 256 \
  --harmbench-limit 50 \
  --output-csv outputs/steered_l1516.csv \
  --output-summary outputs/steered_l1516_summary.json
```

## 3) Live Checkpoints

Judge endpoint readiness:

```bash
nc -z localhost 8002 && echo "judge up"
```

GPU monitoring:

```bash
nvidia-smi -l 1
```

Logs:

```bash
tail -f logs/judge.log logs/steer_results.log
```

## 4) Expected Outputs

| Path | Meaning |
|---|---|
| `vectors/refusal_vector_layer*.pt` | Extracted steering vectors per layer |
| `outputs/steered_results.csv` | Per-example results (`behavior`, `response`, `is_jailbroken`) |
| `outputs/steered_summary.json` | Run metadata + final ASR |
| `logs/judge.log` | vLLM judge server logs |
| `logs/steer_results.log` | Steering script console output |

Done when:
- launcher prints completion,
- summary JSON exists with an `asr` field,
- results CSV has rows.

## 5) Troubleshooting

| Symptom | Check | Fix |
|---|---|---|
| Port `8002` not opening | `nc -z localhost 8002` | wait 1-2 mins; inspect `logs/judge.log` |
| CUDA OOM while judge starts | `logs/judge.log` | lower `JUDGE_GPU_UTIL` in `scripts/launch_steering.sh` |
| Stale process/port lock | `fuser 8002/tcp` | `pkill -f vllm.entrypoints.openai.api_server && fuser -k 8002/tcp` |
| Missing outputs | `ls -lh outputs vectors` | ensure `data/refusal_CAA_training.json` exists and rerun |

## 6) Save Artifacts Before Terminating VM

```bash
scp -r root@<INSTANCE_IP>:~/prompt-optimization/outputs ./
scp -r root@<INSTANCE_IP>:~/prompt-optimization/logs ./
scp -r root@<INSTANCE_IP>:~/prompt-optimization/vectors ./
```
