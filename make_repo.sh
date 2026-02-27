#!binbash

# Define the project name
PROJECT_NAME=adversarial-prompt-training

echo 📂 Creating project $PROJECT_NAME

# 1. Create the directory structure
mkdir -p $PROJECT_NAMEsrcadversary
mkdir -p $PROJECT_NAMEsrcdefense
mkdir -p $PROJECT_NAMEsrcevaluator
mkdir -p $PROJECT_NAMEsrctarget
mkdir -p $PROJECT_NAMEsrctraining
mkdir -p $PROJECT_NAMEsrcdata
mkdir -p $PROJECT_NAMEconfigs
mkdir -p $PROJECT_NAMEtests
mkdir -p $PROJECT_NAMEnotebooks
mkdir -p $PROJECT_NAMEscripts

# 2. Initialize uv project
cd $PROJECT_NAME
uv init --lib

# 3. Create Python files (empty or with basic __init__)
touch srcadversarypolicy.py
touch srcdefensegepa_wrapper.py
touch srcevaluatorjudge.py
touch srctargetmodel.py
touch srctrainingloop.py
touch srcdataharmbench_loader.py
touch configsdefault.yaml
touch run_experiment.py
touch .gitignore

# Add __init__.py files to make them proper packages
find src -type d -exec touch {}__init__.py ;

# 4. Add dependencies using uv
echo 📦 Adding dependencies via uv...
uv add torch transformers peft unsloth dspy-ai pyyaml tqdm datasets accelerate bitsandbytes

# 5. Create a basic README
cat EOT  README.md
# Adversarial Prompt Training (REINFORCE + GEPA)

This project explores the co-evolution of LLM jailbreaking (Red Team) and automated defense (Blue Team).

## Setup
1. Install uv `curl -LsSf httpsastral.shuvinstall.sh  sh`
2. Sync environment `uv sync`
3. Run `uv run run_experiment.py`
EOT

# 6. Setup .gitignore for Python
cat EOT  .gitignore
.venv
__pycache__
.pyc
.env
outputs
checkpoints
EOT

echo ✅ Done! To start, run
echo cd $PROJECT_NAME && uv sync