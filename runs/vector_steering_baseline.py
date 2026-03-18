#!/usr/bin/env python3
"""Activation-addition vector steering baseline run.

This runner mirrors project conventions used by other scripts in `runs/`:
- shared defaults via `src.runtime.defaults.load_default_config`
- shared dataset loading via `src.data.load_harmbench_subset`
- shared runtime sessions via `src.runtime.RuntimeCatalog`
- artifact persistence + run manifest via `src.artifacts`
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.artifacts import log_saved_artifacts, write_run_manifest
from src.data import load_harmbench_subset
from src.evaluators import is_yes_verdict
from src.runtime import (
    GenerationRequest,
    GenerationSession,
    HarmbenchJudgeConfig,
    LocalHFConfig,
    RuntimeCatalog,
    TargetModelConfig,
    resolve_hf_token,
)
from src.runtime.defaults import load_default_config
from src.steering_methods.activation_add import ActivationAddition, extract_steering_vectors
from src.types import RunManifest


def resolve_device(device_override: str | None) -> str:
    """Resolve runtime device, preferring explicit override."""
    if device_override:
        return device_override
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for vector steering baseline run."""
    defaults = load_default_config()
    global_defaults = defaults["global"]
    model_defaults = defaults["runtime"]["models"]
    run_defaults = defaults["runs"]["vector_steering_baseline"]

    parser = argparse.ArgumentParser(description="Run activation-addition vector steering baseline on HarmBench.")
    parser.add_argument("--dataset-name", default=global_defaults["dataset_name"])
    parser.add_argument("--dataset-config", default=global_defaults["dataset_config"])
    parser.add_argument("--dataset-split", default=global_defaults["dataset_split"])
    parser.add_argument("--train-size", type=int, default=run_defaults["train_size"])
    parser.add_argument("--val-size", type=int, default=run_defaults["val_size"])
    parser.add_argument("--seed", type=int, default=global_defaults["seed"])
    parser.add_argument("--device", default=global_defaults["device"], help="Device override (e.g. cuda, cpu).")

    parser.add_argument("--target-model-name", default=model_defaults["target_model_name"])
    parser.add_argument("--judge-model-id", default=model_defaults["judge_model_id"])
    parser.add_argument("--seed-prompt", default=run_defaults["seed_prompt"])

    parser.add_argument("--caa-data-path", default=run_defaults["caa_data_path"])
    parser.add_argument("--target-layers", nargs="+", type=int, default=run_defaults["target_layers"])
    parser.add_argument("--steering-coefficient", type=float, default=run_defaults["steering_coefficient"])
    parser.add_argument("--max-new-tokens", type=int, default=run_defaults["max_new_tokens"])
    parser.add_argument("--temperature", type=float, default=run_defaults["temperature"])
    parser.add_argument("--show-progress", action="store_true")

    parser.add_argument("--results-dir", default=run_defaults["results_dir"])
    parser.add_argument("--vector-dir", default=run_defaults["vector_dir"])
    parser.add_argument("--results-csv-name", default=run_defaults["results_csv_name"])
    parser.add_argument("--summary-json-name", default=run_defaults["summary_json_name"])
    return parser.parse_args()


def load_target_model(cfg: TargetModelConfig) -> GenerationSession:
    """Create the local HF target-model generation session."""
    runtime_cfg = LocalHFConfig(
        model_id=cfg.model_id,
        use_4bit=True,
        max_new_tokens=cfg.max_new_tokens,
    )
    return RuntimeCatalog.build_target_session(runtime_cfg)


def _extract_question_stem(question: str) -> str:
    if "\n\nChoices:\n" in question:
        return question.split("\n\nChoices:\n", 1)[0].strip()
    return question.strip()


def _extract_choices(question: str) -> dict[str, str]:
    choices_text = question.split("\n\nChoices:\n", 1)[1] if "\n\nChoices:\n" in question else question
    matches = re.findall(r"\(([A-Z])\)\s*(.*?)(?=\n\([A-Z]\)\s*|$)", choices_text, flags=re.DOTALL)
    return {label: text.strip() for label, text in matches}


def _normalize_label(label: str) -> str:
    normalized = (label or "").strip().upper()
    if normalized.startswith("(") and normalized.endswith(")"):
        normalized = normalized[1:-1]
    return normalized


def build_caa_pairs(caa_path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with caa_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    pos_data: list[dict[str, str]] = []
    neg_data: list[dict[str, str]] = []
    skipped = 0

    for row in data:
        question = str(row.get("question", ""))
        pos_label = _normalize_label(str(row.get("answer_matching_behavior", "")))
        neg_label = _normalize_label(str(row.get("answer_not_matching_behavior", "")))
        choices = _extract_choices(question)
        instruction = _extract_question_stem(question)

        pos_response = choices.get(pos_label)
        neg_response = choices.get(neg_label)
        if not instruction or not pos_response or not neg_response:
            skipped += 1
            continue

        pos_data.append({"instruction": instruction, "response": pos_response})
        neg_data.append({"instruction": instruction, "response": neg_response})

    if not pos_data or not neg_data:
        raise ValueError(f"Failed to build non-empty pos/neg CAA pairs from {caa_path}.")

    print(f"Built CAA pairs: pos={len(pos_data)}, neg={len(neg_data)}, skipped={skipped}")
    return pos_data, neg_data


def evaluate_steered_model(
    examples: list[dict[str, Any]],
    target_session: GenerationSession,
    judge_session: GenerationSession,
    system_prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    show_progress: bool,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Generate responses on val set, judge jailbreaks, and aggregate metrics."""
    iterator = tqdm(examples, desc="Evaluating steered model", disable=not show_progress)
    rows: list[dict[str, Any]] = []
    behaviors: list[str] = []
    generations: list[str] = []
    latencies: list[float] = []

    for ex in iterator:
        behavior = str(ex["prompt"])
        start = time.perf_counter()
        response = target_session.generate(
            GenerationRequest(
                system_prompt=system_prompt,
                user_prompt=behavior,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            ),
            device=device,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        behaviors.append(behavior)
        generations.append(response)
        latencies.append(elapsed_ms)

    verdicts = judge_session.judge(behaviors=behaviors, generations=generations)
    jailbreak_flags = [is_yes_verdict(v) for v in verdicts]

    for behavior, response, verdict, is_jb, latency in zip(
        behaviors,
        generations,
        verdicts,
        jailbreak_flags,
        latencies,
        strict=True,
    ):
        rows.append(
            {
                "behavior": behavior,
                "response": response,
                "judge_verdict": verdict,
                "is_jailbroken": bool(is_jb),
                "latency_ms": float(latency),
            }
        )

    n = len(rows)
    success_count = sum(1 for flag in jailbreak_flags if flag)
    asr = (success_count / n) if n else 0.0
    metrics = {
        "asr": float(asr),
        "refusal_rate": float(1.0 - asr),
        "n_samples": float(n),
        "success_count": float(success_count),
        "latency_ms_mean": float(sum(latencies) / len(latencies)) if latencies else 0.0,
    }
    return metrics, pd.DataFrame(rows)


def save_artifacts(
    *,
    results_dir: Path,
    vector_dir: Path,
    results_df: pd.DataFrame,
    summary_payload: dict[str, Any],
    runtime_profile: str,
    seed: int,
    args: argparse.Namespace,
    run_seconds: float,
) -> None:
    """Persist CSV/JSON artifacts and run manifest."""
    results_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)

    results_csv_path = results_dir / args.results_csv_name
    summary_json_path = results_dir / args.summary_json_name

    results_df.to_csv(results_csv_path, index=False)
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    manifest = RunManifest(
        mode="vector_steering_baseline",
        runtime_profile=runtime_profile,
        seed=seed,
        dataset={
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "dataset_split": args.dataset_split,
            "train_size": args.train_size,
            "val_size": args.val_size,
        },
        models={
            "target_model_name": args.target_model_name,
            "judge_model_id": args.judge_model_id,
        },
        budget={
            "max_new_tokens": args.max_new_tokens,
            "run_seconds": run_seconds,
        },
        extra={
            "target_layers": args.target_layers,
            "selected_layer": args.target_layers[0],
            "steering_coefficient": args.steering_coefficient,
            "caa_data_path": args.caa_data_path,
            "results_csv_path": str(results_csv_path),
            "summary_json_path": str(summary_json_path),
            "vector_dir": str(vector_dir),
            "asr": summary_payload["asr"],
        },
    )
    manifest_path = write_run_manifest(results_dir=results_dir, payload=manifest)

    log_saved_artifacts(
        [
            vector_dir,
            results_csv_path,
            summary_json_path,
            manifest_path,
        ]
    )


def main() -> None:
    run_start = time.time()
    args = parse_args()
    defaults = load_default_config()
    runtime_profile = defaults["global"]["runtime_profile"]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    results_dir = Path(args.results_dir).resolve()
    vector_dir_arg = Path(args.vector_dir)
    vector_dir = vector_dir_arg if vector_dir_arg.is_absolute() else (results_dir / vector_dir_arg)
    vector_dir = vector_dir.resolve()
    caa_data_path = Path(args.caa_data_path).resolve()

    print(f"Loading target model: {args.target_model_name}")
    target_cfg = TargetModelConfig(model_id=args.target_model_name, max_new_tokens=args.max_new_tokens)
    target_session = load_target_model(target_cfg)
    target_model = target_session.runtime.model
    target_tokenizer = target_session.runtime.tokenizer

    print(f"Loading judge model: {args.judge_model_id}")
    judge_session = RuntimeCatalog.build_judge_session(HarmbenchJudgeConfig(model_id=args.judge_model_id))

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
    print(f"Loaded HarmBench dataset: train={len(train_data)}, val={len(val_data)}")

    print(f"Building CAA pairs from: {caa_data_path}")
    pos_data, neg_data = build_caa_pairs(caa_data_path)

    print(
        "Extracting steering vectors with "
        f"layers={args.target_layers}, coefficient={args.steering_coefficient}"
    )
    vectors = extract_steering_vectors(
        model=target_model,
        tokenizer=target_tokenizer,
        pos_data=pos_data,
        neg_data=neg_data,
        target_layers=args.target_layers,
    )

    vector_dir.mkdir(parents=True, exist_ok=True)
    for layer_idx, vector in vectors.items():
        save_path = vector_dir / f"refusal_vector_layer{layer_idx}.pt"
        torch.save(vector, save_path)
        print(f"Saved steering vector: {save_path}")

    selected_layer = args.target_layers[0]
    steerer = ActivationAddition(target_model, target_tokenizer)
    steerer.apply(
        layer_idx=selected_layer,
        steering_vector=vectors[selected_layer],
        coefficient=args.steering_coefficient,
    )

    try:
        metrics, results_df = evaluate_steered_model(
            examples=val_data,
            target_session=target_session,
            judge_session=judge_session,
            system_prompt=args.seed_prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            show_progress=args.show_progress,
        )
    finally:
        steerer.remove()

    summary_payload = {
        "target_model_name": args.target_model_name,
        "judge_model_id": args.judge_model_id,
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "dataset_size": int(metrics["n_samples"]),
        "target_layers": args.target_layers,
        "selected_layer": selected_layer,
        "steering_coefficient": args.steering_coefficient,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "asr": metrics["asr"],
        "refusal_rate": metrics["refusal_rate"],
        "latency_ms_mean": metrics["latency_ms_mean"],
    }

    run_seconds = time.time() - run_start
    save_artifacts(
        results_dir=results_dir,
        vector_dir=vector_dir,
        results_df=results_df,
        summary_payload=summary_payload,
        runtime_profile=runtime_profile,
        seed=args.seed,
        args=args,
        run_seconds=run_seconds,
    )

    print("Vector steering run complete.")
    print(f"ASR: {summary_payload['asr']:.4f}")
    print(f"Refusal rate: {summary_payload['refusal_rate']:.4f}")


if __name__ == "__main__":
    main()
