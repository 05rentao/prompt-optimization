#!/usr/bin/env python3
"""Adversary-only fine-tuning runner using dataset prompts and model judgment.

This script intentionally excludes prompt-optimization steps (for example GEPA or
reflection loops). It trains only the adversary model weights with REINFORCE
signals from target responses judged by either HarmBench judge verdicts or the
shared heuristic evaluator.
"""

from __future__ import annotations


import argparse
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as f

from src.artifacts import (
    build_baseline_optimized_df,
    log_saved_artifacts,
    save_adversary_asr_vs_iterations_plot,
    save_adversary_refusal_vs_iterations_plot,
    save_baseline_optimized_plot,
    write_json,
    write_many_csv,
    write_run_manifest,
)
from src.data import load_harmbench_subset
from src.evaluators import compute_refusal_score
from src.run_pipeline import (
    adversary_rewrite_sample,
    build_prompt_pool,
    compute_reward_and_verdict,
    maybe_save_adapters as maybe_save_adapters_common,
    split_prompt_pool,
)
from src.runtime import (
    EvaluationConfig,
    EvaluationResult,
    GenerationRequest,
    GenerationSession,
    HarmbenchJudgeConfig,
    ModelConfig,
    RuntimeCatalog,
    UnslothAdversaryConfig,
    build_vllm_target_session,
    cap_thread_workers,
    evaluate_outputs,
    resolve_hf_token,
    resolve_reflection_env_overrides,
    timed_target_generate,
)
from src.runtime.defaults import build_config_snapshot, load_default_config
from src.types import RunManifest


@dataclass
class AdversaryTrainingConfig:
    """Config values controlling adversary-only fine-tuning and eval cadence."""

    iterations: int = 20
    lr: float = 5e-5
    weight_decay: float = 0.01
    eval_every: int = 4
    train_slice_end: int = 50
    eval_slice_start: int = 100
    eval_slice_end: int = 110
    max_new_tokens: int = 120
    target_max_new_tokens: int = 150
    training_csv_name: str = "adversary_training_log.csv"
    eval_csv_name: str = "adversary_eval_outputs.csv"


@dataclass
class RunContext:
    """Long-lived runtime sessions and device state for one script run."""

    adversary_session: GenerationSession
    target_session: GenerationSession
    judge_session: GenerationSession
    device: str


def clean_text(value: Any) -> str:
    """Normalize values for single-line artifact rows."""
    return str(value).replace("\n", " ").replace("\r", " ").strip()


def adversary_output(
    prompt: str,
    attacker_instruction: str,
    adversary_session: GenerationSession,
    device: str,
    max_new_tokens: int,
) -> tuple[dict[str, Any], float]:
    """Generate one rewritten attack prompt from the adversary policy; return elapsed ms."""
    return adversary_rewrite_sample(
        adversary_session,
        device,
        prompt,
        attacker_instruction,
        max_new_tokens=max_new_tokens,
    )


def target_generate(
    prompt: str,
    target_session: GenerationSession,
    device: str,
    target_system_prompt: str,
    max_new_tokens: int = 150,
) -> tuple[str, float]:
    """Generate one target response for a rewritten prompt; return elapsed ms."""

    request = GenerationRequest(
        system_prompt=target_system_prompt,
        user_prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    return timed_target_generate(target_session, device, request)


def pad_gen_ids_batch(
    batch_gen_ids: Sequence[torch.Tensor], pad_id: int
) -> tuple[torch.Tensor, list[int]]:
    """Right-pad (1, L) tensors to a common L for batched forward; keep valid lengths."""
    lengths = [int(x.shape[-1]) for x in batch_gen_ids]
    max_len = max(lengths)
    rows: list[torch.Tensor] = []
    for x in batch_gen_ids:
        if x.dim() != 2 or x.size(0) != 1:
            raise ValueError(f"expected gen_ids shape (1, L), got {tuple(x.shape)}")
        L = x.size(1)
        if L < max_len:
            pad = torch.full(
                (1, max_len - L),
                pad_id,
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat([x, pad], dim=1)
        rows.append(x)
    return torch.cat(rows, dim=0), lengths


def reinforce_update_batch_sgd(
    model: Any,
    optimizer: Any,
    gen_ids: Any,
    prompt_lens: list[int],
    rewards: Any,
    valid_seq_lens: list[int] | None = None,
) -> tuple[float, Any]:
    """Apply a REINFORCE-style SGD step: mean over batch of (-reward_i * sum log pi)."""

    model.train()
    if valid_seq_lens is None:
        valid_seq_lens = [int(gen_ids.size(1))] * int(gen_ids.size(0))

    out = model(input_ids=gen_ids, use_cache=False)
    logits = out.logits
    log_probs = f.log_softmax(logits[:, :-1, :], dim=-1)

    logprob_sums = []
    for i in range(gen_ids.size(0)):
        prompt_len = int(prompt_lens[i])
        valid_end = int(valid_seq_lens[i])
        comp_ids = gen_ids[i, prompt_len:valid_end].clone()
        if comp_ids.numel() == 0:
            logprob_sums.append(torch.tensor(0.0, device=gen_ids.device))
            continue
        start = prompt_len - 1
        end = start + comp_ids.size(0)
        lp = log_probs[i, start:end, :]
        tok_lp = torch.gather(lp, -1, comp_ids.unsqueeze(-1)).squeeze(-1)
        logprob_sums.append(tok_lp.sum())

    logprob_sums = torch.stack(logprob_sums)
    loss = -(rewards * logprob_sums).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu()), logprob_sums.detach()


def reward_from_response(
    prompt: str,
    target_response: str,
    eval_cfg: EvaluationConfig,
    ctx: RunContext,
) -> tuple[float, str]:
    """Compute reward and verdict label from target response quality."""
    return compute_reward_and_verdict(
        behavior=prompt,
        generation=target_response,
        eval_cfg=eval_cfg,
        judge_session=ctx.judge_session if eval_cfg.method == "judge" else None,
    )


def _evaluation_result_to_metrics(ev: EvaluationResult, latency_ms_mean: float) -> dict[str, float]:
    """Align aggregate metric keys with CoEV / ``evaluate_prompts`` JSON exports."""
    aggregate_score = ev.mean_refusal_score
    if aggregate_score is None:
        aggregate_score = ev.refusal_rate
    return {
        "refusal_rate": float(ev.refusal_rate),
        "asr": float(ev.asr),
        "mean_refusal_score": float(ev.mean_refusal_score or 0.0),
        "aggregate_score": float(aggregate_score),
        "latency_ms_mean": float(latency_ms_mean),
    }


def evaluate_asr(
    eval_examples: list[dict[str, Any]],
    attacker_instruction: str,
    target_system_prompt: str,
    train_cfg: AdversaryTrainingConfig,
    eval_cfg: EvaluationConfig,
    ctx: RunContext,
    target_max_workers: int | None = None,
) -> tuple[EvaluationResult, list[dict[str, Any]]]:
    """Run held-out eval prompts and return aggregate metrics plus row payloads.

    Per example the chain is adversary → target, but across examples we **pipeline**:
    after each adversary finishes we submit that target request to a pool while the main
    thread runs the next adversary (Unsloth stays sequential; vLLM can overlap in flight).
    """
    n = len(eval_examples)
    workers = cap_thread_workers(n, target_max_workers)

    use_target_pool = getattr(ctx.target_session.runtime, "supports_concurrent_target_inference", False)

    adv_rows: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    target_texts: list[str] = []
    target_latencies: list[float] = []

    if not use_target_pool:
        for ex in eval_examples:
            sample, adv_latency_ms = adversary_output(
                prompt=ex["prompt"],
                attacker_instruction=attacker_instruction,
                adversary_session=ctx.adversary_session,
                device=ctx.device,
                max_new_tokens=train_cfg.max_new_tokens,
            )
            request = GenerationRequest(
                system_prompt=target_system_prompt,
                user_prompt=sample["completion_text"].strip(),
                max_new_tokens=train_cfg.target_max_new_tokens,
                temperature=0.0,
            )
            target_resp, target_latency_ms = timed_target_generate(ctx.target_session, ctx.device, request)
            adv_rows.append((ex, sample, adv_latency_ms))
            target_texts.append(target_resp)
            target_latencies.append(target_latency_ms)
    else:
        pending: list[tuple[dict[str, Any], dict[str, Any], float, Future[tuple[str, float]]]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for ex in eval_examples:
                sample, adv_latency_ms = adversary_output(
                    prompt=ex["prompt"],
                    attacker_instruction=attacker_instruction,
                    adversary_session=ctx.adversary_session,
                    device=ctx.device,
                    max_new_tokens=train_cfg.max_new_tokens,
                )
                request = GenerationRequest(
                    system_prompt=target_system_prompt,
                    user_prompt=sample["completion_text"].strip(),
                    max_new_tokens=train_cfg.target_max_new_tokens,
                    temperature=0.0,
                )
                fut = executor.submit(timed_target_generate, ctx.target_session, ctx.device, request)
                pending.append((ex, sample, adv_latency_ms, fut))

            for ex, sample, adv_latency_ms, fut in pending:
                adv_rows.append((ex, sample, adv_latency_ms))
                target_resp, target_latency_ms = fut.result()
                target_texts.append(target_resp)
                target_latencies.append(target_latency_ms)

    behaviors: list[str] = []
    generations: list[str] = []
    rows: list[dict[str, Any]] = []
    for (ex, sample, adv_latency_ms), target_response, target_latency_ms in zip(
        adv_rows, target_texts, target_latencies, strict=True
    ):
        prompt = ex["prompt"]
        adversary_text = sample["completion_text"].strip()
        refusal_score = compute_refusal_score(target_response)
        total_latency = adv_latency_ms + target_latency_ms
        behaviors.append(prompt)
        generations.append(target_response)
        rows.append(
            {
                "id": str(ex.get("id", "")),
                "behavior": prompt,
                "generation": target_response,
                "prompt": clean_text(prompt),
                "adversary_prompt": clean_text(adversary_text),
                "target_response": clean_text(target_response),
                "refusal_score": refusal_score,
                "asr_score": 1.0 - refusal_score,
                "adversary_latency_ms": adv_latency_ms,
                "target_latency_ms": target_latency_ms,
                "latency_ms_total": total_latency,
                "latency_ms": total_latency,
            }
        )

    eval_result = evaluate_outputs(
        behaviors=behaviors,
        generations=generations,
        cfg=eval_cfg,
        judge_session=ctx.judge_session if eval_cfg.method == "judge" else None,
    )
    return eval_result, rows


def _eval_suite(
    eval_examples: list[dict[str, Any]],
    attacker_instruction: str,
    target_system_prompt: str,
    train_cfg: AdversaryTrainingConfig,
    eval_cfg: EvaluationConfig,
    ctx: RunContext,
    args: argparse.Namespace,
) -> tuple[EvaluationResult, list[dict[str, Any]], dict[str, float]]:
    """Baseline / periodic / final eval with shared kwargs."""
    ev, rows = evaluate_asr(
        eval_examples=eval_examples,
        attacker_instruction=attacker_instruction,
        target_system_prompt=target_system_prompt,
        train_cfg=train_cfg,
        eval_cfg=eval_cfg,
        ctx=ctx,
        target_max_workers=args.target_max_workers,
    )
    df = pd.DataFrame(rows)
    lat_mean = float(df["latency_ms_total"].mean()) if not df.empty else 0.0
    metrics = _evaluation_result_to_metrics(ev, lat_mean)
    return ev, rows, metrics


def _try_save_adapters_print(ctx: RunContext, save_dir: str | None, results_dir: Path) -> str | None:
    path = maybe_save_adapters_common(ctx.adversary_session, save_dir, results_dir=results_dir)
    if path:
        print(f"Saved adversary adapters to: {path}")
    return path


def save_artifacts(
    args: argparse.Namespace,
    results_dir: Path,
    train_cfg: AdversaryTrainingConfig,
    baseline_metrics: dict[str, float],
    final_metrics: dict[str, float],
    baseline_df: pd.DataFrame,
    final_df: pd.DataFrame,
    training_df: pd.DataFrame,
    run_seconds: float,
    adapter_path: str | None,
    config_snapshot: dict[str, Any],
) -> None:
    """Persist metrics, CSVs, plots, and manifest.

    Prompts are fixed for the run (from CLI/config); artifacts compare **adversary
    weights** before vs after REINFORCE, not prompt optimization. Instruction and
    target system prompt text are not written to disk.
    """

    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload: dict[str, Any] = {
        "config": {
            "mode": args.mode,
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "dataset_split": args.dataset_split,
            "train_size": args.train_size,
            "val_size": args.val_size,
            "runtime_profile": args.runtime_profile,
            "target_model_name": args.task_model_name,
            "reflection_vllm_base_url": args.reflection_vllm_base_url,
            "eval_method": args.eval_method,
            "iterations": train_cfg.iterations if args.mode == "train" else 0,
            "run_seconds": run_seconds,
        },
        "baseline_metrics": baseline_metrics,
        "final_metrics": final_metrics,
    }
    metrics_json_path = results_dir / "adversary_run_metrics.json"
    write_json(metrics_json_path, metrics_payload)

    comparison_df = build_baseline_optimized_df(
        baseline_metrics=baseline_metrics,
        optimized_metrics=final_metrics,
        baseline_variant="before_training",
        comparison_variant="after_training",
    )
    write_many_csv(
        results_dir,
        {
            "eval_metrics_before_vs_after_training.csv": comparison_df,
            "eval_outputs_before_training.csv": baseline_df,
            "eval_outputs_after_training.csv": final_df,
            train_cfg.training_csv_name: training_df,
        },
        skip_empty={train_cfg.training_csv_name},
    )

    plot_path = save_baseline_optimized_plot(
        comparison_df=comparison_df,
        out_path=results_dir / "plot_eval_metrics_before_vs_after_training.png",
        title="Adversary eval: before vs after training",
        subtitle=(
            "Same fixed rewriter instruction and target system prompt. "
            "before_training = adversary LoRA at init; after_training = same LoRA after REINFORCE. "
            "Bars = aggregate metrics on the held-out eval set (not per-step noise)."
        ),
    )
    train_iters = train_cfg.iterations if args.mode == "train" else None
    final_asr_train = final_metrics.get("asr") if args.mode == "train" else None
    final_refusal_train = (1.0 - float(final_asr_train)) if final_asr_train is not None else None
    asr_iter_path = save_adversary_asr_vs_iterations_plot(
        training_df=training_df,
        out_path=results_dir / "plot_asr_vs_iterations.png",
        title="Adversary ASR vs training iteration",
        final_asr=final_asr_train,
        iterations=train_iters,
    )
    refusal_iter_path = save_adversary_refusal_vs_iterations_plot(
        training_df=training_df,
        out_path=results_dir / "plot_refusal_vs_iterations.png",
        title="Adversary refusal rate vs training iteration",
        final_refusal=final_refusal_train,
        iterations=train_iters,
    )

    manifest = RunManifest(
        mode=f"adversary_{args.mode}",
        runtime_profile=args.runtime_profile,
        seed=args.seed,
        dataset={
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "dataset_split": args.dataset_split,
            "train_size": args.train_size,
            "val_size": args.val_size,
        },
        models={
            "adversary_model": args.adversary_model_id,
            "target_model": args.task_model_name,
            "judge_model": args.judge_model_id,
        },
        budget={
            "iterations": train_cfg.iterations if args.mode == "train" else 0,
            "eval_every": train_cfg.eval_every if args.mode == "train" else 0,
            "run_seconds": run_seconds,
        },
        endpoints={"reflection_base_url": args.reflection_vllm_base_url},
        extra={
            "metrics_json_path": str(metrics_json_path),
            "comparison_metrics_csv": "eval_metrics_before_vs_after_training.csv",
            "training_csv_name": train_cfg.training_csv_name,
            "eval_csv_name": train_cfg.eval_csv_name,
            "save_dir": adapter_path,
            "eval_method": args.eval_method,
        },
        config_snapshot=config_snapshot,
    )
    manifest_path = write_run_manifest(results_dir=results_dir, payload=manifest)
    logged_paths = [
        metrics_json_path,
        manifest_path,
        plot_path,
        results_dir,
    ]
    if asr_iter_path is not None:
        logged_paths.insert(-1, asr_iter_path)
    if refusal_iter_path is not None:
        logged_paths.insert(-1, refusal_iter_path)
    log_saved_artifacts(logged_paths)


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    """Parse CLI flags for adversary-only fine-tuning and evaluation."""
    global_defaults = defaults["global"]
    run_defaults = defaults["runs"]["adversary"]
    parser = argparse.ArgumentParser(description="Run adversary-only fine-tuning (no prompt optimization).")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--device", default=global_defaults["device"], help="Device override (e.g. cuda, cpu).")
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Optional directory for adversary adapters; relative paths are under --results-dir.",
    )

    parser.add_argument("--dataset-name", default=global_defaults["dataset_name"])
    parser.add_argument("--dataset-config", default=global_defaults["dataset_config"])
    parser.add_argument("--dataset-split", default=global_defaults["dataset_split"])
    parser.add_argument("--train-size", type=int, default=run_defaults["train_size"])
    parser.add_argument("--val-size", type=int, default=run_defaults["val_size"])
    parser.add_argument("--seed", type=int, default=global_defaults["seed"])

    parser.add_argument("--results-dir", default=run_defaults["results_dir"])
    parser.add_argument(
        "--target-max-workers",
        type=int,
        default=run_defaults.get("target_max_workers", 16),
        help=(
            "Thread-pool size for target HTTP calls during eval; pipelines with adversary "
            "(next adversary runs while earlier targets are in flight)."
        ),
    )
    parser.add_argument("--eval-method", choices=["judge", "heuristic"], default=run_defaults["eval_method"])
    parser.add_argument("--refusal-threshold", type=float, default=run_defaults["refusal_threshold"])
    parser.add_argument("--asr-threshold", type=float, default=run_defaults["asr_threshold"])
    parser.add_argument("--attacker-instruction", default=run_defaults["attacker_instruction"])
    parser.add_argument("--target-system-prompt", default=run_defaults["target_system_prompt"])
    parser.add_argument(
        "--adversary-reinforce-batch-size",
        type=int,
        default=run_defaults.get("adversary_reinforce_batch_size", 1),
        help="K adversary→target rollouts per REINFORCE step (padded batch, mean loss).",
    )
    return parser.parse_args()


def _patch_args_from_yaml(args: argparse.Namespace, defaults: dict[str, Any]) -> None:
    """Attach runtime/model fields from YAML (exposed as args.* for manifest/metrics)."""
    args.runtime_profile = defaults["global"]["runtime_profile"]
    models = defaults["runtime"]["models"]
    args.adversary_model_id = models["adversary_model_id"]
    args.task_model_name = models["target_model_name"]
    args.judge_model_id = models["judge_model_id"]
    rw_url, rw_key = resolve_reflection_env_overrides(defaults)
    args.reflection_vllm_base_url = rw_url
    args.reflection_vllm_api_key = rw_key


def _build_context(args: argparse.Namespace, defaults: dict[str, Any], device: str) -> RunContext:
    model_cfg = ModelConfig(model_id=args.adversary_model_id)
    adversary_cfg = UnslothAdversaryConfig(
        model_id=model_cfg.model_id,
        max_seq_length=model_cfg.max_seq_length,
        load_in_4bit=model_cfg.load_in_4bit,
        lora_r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
    )
    return RunContext(
        adversary_session=RuntimeCatalog.build_adversary_session(adversary_cfg),
        target_session=build_vllm_target_session(defaults),
        judge_session=RuntimeCatalog.build_judge_session(HarmbenchJudgeConfig()),
        device=device,
    )


def main() -> None:
    """Execute adversary-only training/eval and persist run artifacts."""
    run_start = time.time()
    defaults = load_default_config()
    args = parse_args(defaults)
    _patch_args_from_yaml(args, defaults)
    sns.set_theme(style="whitegrid")
    run_defaults = defaults["runs"]["adversary"]

    train_cfg = AdversaryTrainingConfig(
        iterations=run_defaults["iterations"],
        lr=run_defaults["lr"],
        weight_decay=run_defaults["weight_decay"],
        eval_every=run_defaults["eval_every"],
        train_slice_end=run_defaults["train_slice_end"],
        eval_slice_start=run_defaults["eval_slice_start"],
        eval_slice_end=run_defaults["eval_slice_end"],
        max_new_tokens=run_defaults["max_new_tokens"],
        target_max_new_tokens=int(run_defaults.get("target_max_new_tokens", 150)),
        training_csv_name=run_defaults["training_csv_name"],
        eval_csv_name=run_defaults["eval_csv_name"],
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    eval_cfg = EvaluationConfig(
        method=args.eval_method,
        refusal_threshold=args.refusal_threshold,
        asr_threshold=args.asr_threshold,
    )

    ctx = _build_context(args, defaults, device)

    hf_token = resolve_hf_token()
    train_data, val_data, _ = load_harmbench_subset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.dataset_split,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
        hf_token=hf_token,
    )
    prompts = build_prompt_pool(train_data, val_data)
    train_prompts, eval_prompts = split_prompt_pool(
        prompts=prompts,
        train_slice_end=train_cfg.train_slice_end,
        eval_slice_start=train_cfg.eval_slice_start,
        eval_slice_end=train_cfg.eval_slice_end,
        script_name="adversary_run",
    )
    eval_examples = [{"id": f"eval_{i}", "prompt": p} for i, p in enumerate(eval_prompts)]

    baseline_eval, baseline_rows, baseline_metrics = _eval_suite(
        eval_examples,
        args.attacker_instruction,
        args.target_system_prompt,
        train_cfg,
        eval_cfg,
        ctx,
        args,
    )
    print("Baseline metrics:", baseline_metrics)

    training_rows: list[dict[str, Any]] = []
    final_eval = baseline_eval
    final_eval_rows = baseline_rows
    final_metrics = baseline_metrics
    model = ctx.adversary_session.runtime
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    if args.mode == "train":
        batch_size = max(1, int(args.adversary_reinforce_batch_size))
        for iteration in range(train_cfg.iterations):
            idx = torch.randint(0, len(train_prompts), ()).item()
            prompt = train_prompts[idx]

            batch_gen_ids: list[torch.Tensor] = []
            batch_rewards: list[float] = []
            batch_prompt_lens: list[int] = []
            adv_texts: list[str] = []
            target_resps: list[str] = []
            verdicts: list[str] = []

            for _ in range(batch_size):
                sampled, _ = adversary_output(
                    prompt=prompt,
                    attacker_instruction=args.attacker_instruction,
                    adversary_session=ctx.adversary_session,
                    device=ctx.device,
                    max_new_tokens=train_cfg.max_new_tokens,
                )
                adversary_text = sampled["completion_text"].strip()
                target_response, _ = target_generate(
                    prompt=adversary_text,
                    target_session=ctx.target_session,
                    device=ctx.device,
                    target_system_prompt=args.target_system_prompt,
                    max_new_tokens=train_cfg.target_max_new_tokens,
                )
                reward, verdict = reward_from_response(prompt, target_response, eval_cfg, ctx)
                batch_gen_ids.append(sampled["gen_ids"])
                batch_rewards.append(reward)
                batch_prompt_lens.append(sampled["prompt_len"])
                adv_texts.append(adversary_text)
                target_resps.append(target_response)
                verdicts.append(verdict)

            tok = model.tokenizer
            pad_id = tok.pad_token_id
            if pad_id is None:
                pad_id = tok.eos_token_id
            if pad_id is None:
                raise RuntimeError("Tokenizer has no pad_token_id or eos_token_id for batch padding.")
            padded_ids, valid_lens = pad_gen_ids_batch(batch_gen_ids, int(pad_id))
            device_t = padded_ids.device
            rewards_t = torch.tensor(batch_rewards, dtype=torch.float32, device=device_t)
            loss_val, _ = reinforce_update_batch_sgd(
                model=model,
                optimizer=optimizer,
                gen_ids=padded_ids,
                prompt_lens=batch_prompt_lens,
                rewards=rewards_t,
                valid_seq_lens=valid_lens,
            )

            mean_r = float(rewards_t.mean().item())

            eval_asr: float | str = ""
            eval_refusal_rate: float | str = ""
            if iteration % train_cfg.eval_every == 0:
                periodic_eval, _, periodic_metrics = _eval_suite(
                    eval_examples,
                    args.attacker_instruction,
                    args.target_system_prompt,
                    train_cfg,
                    eval_cfg,
                    ctx,
                    args,
                )
                eval_asr = periodic_eval.asr
                eval_refusal_rate = periodic_eval.refusal_rate
                print(
                    f"Iter {iteration} | ASR: {periodic_eval.asr:.3f} | "
                    f"refusal_rate: {periodic_eval.refusal_rate:.3f} | "
                    f"latency_ms_mean: {periodic_metrics['latency_ms_mean']:.1f}"
                )

            training_rows.append(
                {
                    "iteration": iteration,
                    "dataset_index": idx,
                    "prompt": clean_text(prompt),
                    "adversary_output": clean_text(adv_texts[0]),
                    "target_response": clean_text(target_resps[0]),
                    "reward": mean_r,
                    "loss": loss_val,
                    "verdict": verdicts[0],
                    "max_reward": max(batch_rewards),
                    "batch_size": batch_size,
                    "eval_asr": eval_asr,
                    "eval_refusal_rate": eval_refusal_rate,
                }
            )

        final_eval, final_eval_rows, final_metrics = _eval_suite(
            eval_examples,
            args.attacker_instruction,
            args.target_system_prompt,
            train_cfg,
            eval_cfg,
            ctx,
            args,
        )
        print("Final metrics:", final_metrics)

    results_dir = Path(args.results_dir).resolve()
    adapter_path = _try_save_adapters_print(ctx, args.save_dir, results_dir)
    run_seconds = time.time() - run_start
    baseline_df = pd.DataFrame(baseline_rows)
    final_df = pd.DataFrame(final_eval_rows)
    training_df = pd.DataFrame(training_rows)

    save_artifacts(
        args=args,
        results_dir=results_dir,
        train_cfg=train_cfg,
        baseline_metrics=baseline_metrics,
        final_metrics=final_metrics,
        baseline_df=baseline_df,
        final_df=final_df,
        training_df=training_df,
        run_seconds=run_seconds,
        adapter_path=adapter_path,
        config_snapshot=build_config_snapshot(defaults, cli_args=args),
    )


if __name__ == "__main__":
    main()
