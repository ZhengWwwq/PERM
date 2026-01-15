from __future__ import annotations

import argparse, os, json
import wandb
from dotenv import load_dotenv
import numpy as np

from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

from empathy_dataset import EmpathyDataset
from empathy_judge import (
    prepare_prompt_analysis,
    prepare_prompt_response, 
    prepare_prompt_standby,
    batch_generate_hf, 
    batch_generate_oai, 
    batch_generate_vllm, 
)

from utils import (
    replace_none_with_mean, 
    extract_score, 
    calculate_standby_reward,
    calculate_final_scores, 
    set_random_seed, 
    save_args_to_json, 
    save_jsonl, 
    extract_features
)

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Empathy GRPO trainer")

    # Random seed
    p.add_argument("--seed", type=int, default=42)

    # Data / paths
    p.add_argument("--train_data_path", type=str, required=True)
    p.add_argument("--valid_data_path", type=str, required=False)
    p.add_argument("--num_train_samples", type=int, default=None)
    p.add_argument("--num_valid_samples", type=int, default=None)

    p.add_argument("--output_dir",   type=str, required=True)
    p.add_argument("--resume_ckpt_dir", type=str, default=None)

    # Base / adapter models
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--peft_r",     type=int,   default=8)
    p.add_argument("--peft_alpha", type=int,   default=16)
    p.add_argument("--peft_dropout", type=float, default=0.1)
    p.add_argument("--target_modules",
                   type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # GRPO specific
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)

    # Optim & schedule
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--save_steps", type=int, default=20)
    p.add_argument("--save_total_limit", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--max_completion_length", type=int, default=1024)

    # Reward model
    p.add_argument("--reward_model_type", type=str, default="oai")
    p.add_argument("--reward_generation_kwargs", type=json.loads, default="{}")
    p.add_argument("--calculate_score_mode", type=str, default="mean")
    p.add_argument("--calculate_score_weights", type=json.loads, default="[]")
    p.add_argument("--use_length_penalty", action="store_true", default=False)
    p.add_argument("--length_penalty_limit", type=int, default=300)
    p.add_argument("--length_penalty_coef", type=float, default=0.001)
    p.add_argument("--use_standby_reward", action="store_true", default=False)
    p.add_argument("--standby_reward_coef", type=float, default=0.1)
    
    # Precision / hardware
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_lora", action="store_true", default=False)
    p.add_argument("--use_vllm", action="store_true", default=False)
    p.add_argument("--vllm_mode", type=str, choices=["serve", "colocate"], default="serve")

    # Tracking
    p.add_argument("--wandb_project", type=str)
    p.add_argument("--wandb_entity",  type=str)

    # Optional JSON/YAML override
    p.add_argument("--config_file", type=str)

    args = p.parse_args()
    if args.config_file:
        with open(args.config_file) as f:
            override = json.load(f) if args.config_file.endswith(".json") else \
                       __import__("yaml").safe_load(f)
        for k, v in override.items():
            setattr(args, k, v)

    return args



# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_random_seed(args.seed)
    save_args_to_json(args, args.output_dir)

    # Dataset
    train_dataset = EmpathyDataset(args.train_data_path, args.num_train_samples)
    valid_dataset = EmpathyDataset(args.valid_data_path, args.num_valid_samples) if args.valid_data_path else None

    # LoRA
    lora_cfg = LoraConfig(
        r=args.peft_r,
        lora_alpha=args.peft_alpha,
        task_type="CAUSAL_LM",
        target_modules=args.target_modules.split(","),
    ) if args.use_lora else None

    # W&B
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.output_dir.replace("/", "_"),
            config=vars(args),
            save_code=True,
            job_type="train",
        )

    # GRPO Config
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, 
        warmup_ratio=args.warmup_ratio,
        beta=args.beta,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.epochs,
        num_generations=args.num_generations,
        max_prompt_length=1024,
        max_completion_length=args.max_completion_length,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if valid_dataset else 'no', 
        eval_steps=args.eval_steps if valid_dataset else None,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode
    )

    # --------------------------------------------------------------------------- #
    # Reward Functions
    # --------------------------------------------------------------------------- #

    rewards_logging = {"resonation": [], "expression": [], "reception": [], "standby": []}
    predict_logging = []

    # Length penalty
    def length_penalty_reward_func(completion_ids, **kwargs):
        rewards = []
        for ids in completion_ids:
            excess_length = max(0, len(ids) - args.length_penalty_limit)
            penalty = excess_length * args.length_penalty_coef
            rewards.append(-penalty) 
        return rewards
    
    def standby_reward_func(prompts, completions, query, scenario, persona, **kwargs):
        # Judge standby reward
        _, all_responses = extract_features(all_completions=completions)
        prompts = [
            prepare_prompt_standby(
                scenario=sc, 
                persona=pa, 
                query=q, 
                response=r
            )
            for sc, pa, q, r in zip(scenario, persona, query, all_responses)
        ]

        outputs = batch_generate_oai(
            prompts=prompts,
            model=args.reward_generation_kwargs['model_name'], 
            api_key=args.reward_generation_kwargs['api_key'],
            base_url=args.reward_generation_kwargs['base_url'],
        )

        all_scores = replace_none_with_mean([calculate_standby_reward(out) for out in outputs])
        rewards_logging["standby"].append(np.mean(all_scores))
        rewards = [args.standby_reward_coef * (score / 100. - 1.0) for score in all_scores]
        return rewards

    if args.reward_model_type == "hf":
        def empathy_reward_hf(prompts, completions, query, scenario, persona, **kwargs):
            """
            Empathy reward function using Hugging Face model.
            """
            raise ValueError("Not implemented yet")
        reward_func = empathy_reward_hf

    elif args.reward_model_type == "oai":
        def empathy_reward_oai(prompts, completions, query, scenario, persona, **kwargs):
            """
            Empathy reward function using OpenAI model.
            """
            all_analysis, all_responses = extract_features(all_completions=completions)
            judge_scores = {}
            # Judge analysis
            for dimension in ["resonation"]:
                dim_prompts = [
                    prepare_prompt_analysis(
                        dimension=dimension, 
                        scenario=sc, 
                        persona=pa, 
                        query=q, 
                        analysis=a
                    )
                    for sc, pa, q, a in zip(scenario, persona, query, all_analysis)
                ]
                outputs = batch_generate_oai(
                    prompts=dim_prompts,
                    model=args.reward_generation_kwargs['model_name'], 
                    api_key=args.reward_generation_kwargs['api_key'],
                    base_url=args.reward_generation_kwargs['base_url'],
                )
                judge_scores[dimension] = replace_none_with_mean([extract_score(out) for out in outputs])
                rewards_logging[dimension].append(np.mean(judge_scores[dimension]))
            # Judge response
            for dimension in ["expression", "reception"]:
                dim_prompts = [
                    prepare_prompt_response(
                        dimension=dimension, 
                        scenario=sc, 
                        persona=pa, 
                        query=q, 
                        response=r
                    )
                    for sc, pa, q, r in zip(scenario, persona, query, all_responses)
                ]
                outputs = batch_generate_oai(
                    prompts=dim_prompts,
                    model=args.reward_generation_kwargs['model_name'], 
                    api_key=args.reward_generation_kwargs['api_key'],
                    base_url=args.reward_generation_kwargs['base_url'],
                )
                judge_scores[dimension] = replace_none_with_mean([extract_score(out) for out in outputs])
                rewards_logging[dimension].append(np.mean(judge_scores[dimension]))
            predict_logging.append({'predict': completions[-1], 'prompt': prompts[-1]})
            return calculate_final_scores(judge_scores, cal_mode=args.calculate_score_mode, weights=args.calculate_score_weights)
        reward_func = empathy_reward_oai
    
    elif args.reward_model_type == "vllm":
        def empathy_reward_vllm(prompts, completions, query, scenario, persona, **kwargs):
            """
            Empathy reward function using vLLM model.
            """
            raise ValueError("Not implemented yet")
        reward_func = empathy_reward_vllm

    elif args.reward_model_type == "reward_model":
        reward_func = args.reward_generation_kwargs['model_path']

    if args.use_length_penalty:
        reward_func = [reward_func, length_penalty_reward_func]

    if args.use_standby_reward:
        if isinstance(reward_func, list):
            reward_func.append(standby_reward_func)
        else:
            reward_func = [reward_func, standby_reward_func]

    # GRPO Trainer
    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=reward_func, 
        args=grpo_config, 
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        peft_config=lora_cfg,
    )

    # Train!
    if args.resume_ckpt_dir:
        trainer.train(resume_from_checkpoint=args.resume_ckpt_dir)
    else:
        trainer.train()

    if args.wandb_project:
        wandb.finish()
    
    with open(os.path.join(args.output_dir, "scores.json"), "w") as f:
        json.dump(rewards_logging, f, indent=2)
    save_jsonl(predict_logging, os.path.join(args.output_dir, "predict.json"))

    print("*" * 20 + " Finish Training!!! " + "*" * 20)

if __name__ == "__main__":
    load_dotenv(".env")
    main()