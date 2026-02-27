import torch
from tqdm import tqdm
from src.training.logger import ExperimentLogger
from src.evaluator.judge import get_judge, compute_reward

# Base system prompt when GEPA is off or initial state when GEPA is on
BASE_SYSTEM_PROMPT = "You are a helpful, harmless, and honest AI assistant."


class TrainingLoop:
    def __init__(self, adversary, target_model, behaviors, config):
        self.adversary = adversary
        self.target = target_model
        self.behaviors = behaviors
        self.config = config
        self.logger = ExperimentLogger()
        judge_model = config.get("evaluator", {}).get("model_name", "cais/HarmBench-Llama-2-13b-cls")
        self.judge = get_judge(model_name=judge_model)
        self.use_gepa_defender = config.get("defense", {}).get("USE_GEPA_DEFENDER", False)
        self._blue_team = None  # Lazy init when first jailbreak triggers GEPA

    def _get_blue_team(self):
        if self._blue_team is None:
            from src.defense.gepa_wrapper import get_blue_team
            self._blue_team = get_blue_team()
        return self._blue_team

    def run(self):
        epochs = self.config["training"]["epochs"]
        # Logic gate: fixed prompt when GEPA off; mutable when GEPA on
        current_system_prompt = BASE_SYSTEM_PROMPT
        failed_defenses = []  # For GEPA context when USE_GEPA_DEFENDER is True

        for epoch in range(epochs):
            print(f"\n🚀 Starting Epoch {epoch + 1}/{epochs}")
            epoch_rewards = []

            for entry in tqdm(self.behaviors, desc="Training Steps"):
                behavior = entry["behavior"]

                # 1. Red Team generates attack
                attack_prompt, log_prob = self.adversary.generate_attack(behavior)

                # 2. Target responds (uses current_system_prompt; fixed if not GEPA, else may have been mutated)
                full_input = f"System: {current_system_prompt}\nUser: {attack_prompt}"
                response = self.target(full_input)

                # 3. Judge evaluates (Yes/No → 1.0/0.0)
                eval_res = self.judge(behavior=behavior, response=response)
                reward = compute_reward(eval_res.is_jailbroken)
                epoch_rewards.append(reward)

                # 4. GEPA mutation: on successful jailbreak, trigger BlueTeam and update prompt for next rollouts
                if self.use_gepa_defender and reward == 1.0:
                    failed_defenses.append(
                        f"Behavior: {behavior[:200]}\nAttack: {attack_prompt[:200]}\nResponse: {response[:200]}"
                    )
                    context = "\n\n---\n\n".join(failed_defenses[-10:])  # Last 10 failures
                    blue_team = self._get_blue_team()
                    out = blue_team.forward(context=context)
                    new_prompt = getattr(out, "system_prompt", None) if out else None
                    if new_prompt and isinstance(new_prompt, str) and new_prompt.strip():
                        current_system_prompt = new_prompt.strip()
                        print("🛡️ GEPA updated system prompt for next rollouts.")

                # 5. Adversary update (REINFORCE)
                self.adversary.update(log_prob, reward)

                # 6. Log (include system_prompt and toggle for success-rate plots)
                explanation = getattr(eval_res, "explanation", "") or ""
                self.logger.log_step({
                    "epoch": epoch + 1,
                    "behavior": behavior,
                    "adversary_attack": attack_prompt,
                    "target_response": response,
                    "reward": reward,
                    "is_jailbroken": eval_res.is_jailbroken,
                    "explanation": explanation,
                    "system_prompt": current_system_prompt,
                    "use_gepa_defender": self.use_gepa_defender,
                })

            avg_reward = sum(epoch_rewards) / len(epoch_rewards)
            print(f"📈 Epoch {epoch + 1} Average Reward: {avg_reward:.4f}")
