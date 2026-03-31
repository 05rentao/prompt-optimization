#!/bin/bash
# Run this once on a fresh Prime Intellect node to set up the environment.

# Install miniconda if not present
if [ ! -f ~/miniconda3/bin/conda ]; then
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b
  ~/miniconda3/bin/conda init bash
  source ~/.bashrc
fi

# Create env if it doesn't exist, otherwise just activate
if conda env list | grep -q promptopt; then
  echo "Environment already exists, activating..."
else
  echo "Creating new environment..."
  conda create -n promptopt python=3.11 -y
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate promptopt

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install vllm==0.7.3
pip install transformers==4.48.2 accelerate datasets
pip install unsloth
pip install matplotlib seaborn pandas numpy openai tqdm scikit-learn pyyaml requests litellm peft typing_extensions gepa

echo "Done! Environment ready. Run: conda activate promptopt"
