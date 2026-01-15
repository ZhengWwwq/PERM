# PERM: Psychology-grounded Empathetic Reward Modeling for Large Language Models

[![Datasets & Models](https://img.shields.io/badge/Datasets_&_Models-HuggingFace-yellow?style=plastic&logo=Hugging%20Face)](https://huggingface.co/Zwwwwwq/PERM-Qwen2.5-7B-Instruct)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red?style=plastic&logo=arxiv)](https://arxiv.org/)

------

## 📌 Overview

**PERM** (Psychology-grounded Empathy Reward Modeling) is a multi-perspective reward modeling framework designed to enhance empathetic behaviors in LLMs.

By jointly modeling empathy from the perspectives of the **empathy seeker**, **empathy supporter**, and **bystander**, PERM provides fine-grained, psychologically grounded supervision signals that go beyond outcome-only or surface-level evaluations.

------

## 🚀 Quick Start

### 🔧 Installation

You can set up the environment as follows:

```bash
conda create -n perm python=3.10
conda activate perm
pip install -r requirements.txt
```

------

### 🧠 PERM Training

To fine-tune a base LLM with the PERM framework:

1. Configure the training hyperparameters in `perm/grpo.sh`.
2. Launch training with:

```bash
cd perm
bash grpo_train.sh
```

The training pipeline is built on **GRPO-style reinforcement learning**, integrating multi-perspective empathy rewards to guide policy optimization.

------

### 📊 Evaluation

We mainly evaluate empathetic performance using **EQ-Bench3**, a widely adopted benchmark for emotional intelligence and empathy in LLMs.

Please refer to the official repository for setup and evaluation details:
👉 [EQ-Bench3](https://github.com/EQ-bench/eqbench3)



