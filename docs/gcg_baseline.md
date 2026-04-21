# Implementation Guide: AttnGCG on Prime Intellect H100


This document outlines the comprehensive process for deploying the AttnGCG (Attention Manipulation) jailbreaking pipeline on your Prime Intellect H100 GPU instance. It covers environment configuration, path mapping for your Llama-3.1 and Qwen2.5 models, execution of the algorithm, and result retrieval.

Reference Github:
[link to OG github](https://github.com/UCSC-VLAA/AttnGCG-attack/blob/main/README.md)

## 1. Prime Intellect Environment Setup
Your current H100 80GB instance with 185GB RAM is the optimal environment for this pipeline, as it meets and exceeds the NVIDIA A100 80GB requirements specified by the authors.

### Step 1: FastChat Installation (Source)
The pipeline requires a specific version of FastChat. You must install it from the source repository rather than via pip.

```bash
git clone [https://github.com/lm-sys/FastChat.git](https://github.com/lm-sys/FastChat.git)
cd FastChat
pip install -e .
cd ..
```

### Step 2: AttnGCG Installation
Install the AttnGCG package in editable mode and run the preparation script to finalize the environment.

```bash
# Navigate to the AttnGCG root directory
pip install -e .
bash prepare.sh
```

## 2. Model Configuration and Path Mapping
Since you are using custom models (Llama-3.1-8B and Qwen2.5-7B) rather than default Hugging Face cache models, you must manually update the configuration files.

### Modifying Configs
Open `experiments/configs/individual_xxx.py` and `experiments/configs/transfer_xxx.py` and update the following lines to point to your local directories:

```python
config.tokenizer_paths=["/path/to/your/Llama-3.1-8B"]
config.model_paths=["/path/to/your/Llama-3.1-8B"]
```

## 3. Running the AttnGCG Algorithm
To compare the effectiveness of AttnGCG against your existing 52% ASR baseline, execute the direct attack script.

### Execution Commands
* **Direct Attack:** Used to generate adversarial suffixes for specific models.
    ```bash
    cd experiments/bash_scripts
    bash run_direct.sh llama2_chat_7b attngcg 0
    ```
* **Generalizing with AutoDAN:** Integrates AttnGCG optimization with AutoDAN-style prompts.
    ```bash
    bash run_autodan.sh llama2_chat_7b attngcg 0
    ```

## 4. Result Retrieval and Evaluation
The results are typically saved in the `experiments/results` directory. To interpret these findings and compare them with your Co-Evolutionary framework, use the evaluation suite.

### ASR Calculation
Run the evaluation script to get the Attack Success Rate using both keyword detection and GPT-4 judging.

```bash
cd eval
# Syntax: bash eval.sh $model $attack $method
bash eval.sh llama2_chat_7b attngcg direct
```

### Where to Look for Results
* **Adversarial Suffixes:** Check the `.json` or `.pkl` files in the `experiments/results` folder.
* **Attention Heatmaps:** Visualizations generated during the process are saved in the `visualize` or `figures` directories if the logging flags are enabled.
* **Logs:** Check the terminal output or log files for the `attention_loss` values to ensure the optimization is progressing as expected.

## 5. Debugging and Troubleshooting

| Issue | Cause | Resolution |
| :--- | :--- | :--- |
| **CUDA Out of Memory (OOM)** | Batch size too large for 80GB VRAM. | Reduce `batch_size` in `experiments/configs/individual_xxx.py` (Default is often 256). |
| **ModuleNotFoundError: 'fschat'** | FastChat not linked correctly. | Ensure you ran `pip install -e .` inside the FastChat root directory. |
| **Invalid Model Path** | Config file pointing to HF cache instead of local. | Double-check absolute paths in your `config.model_paths`. |
| **Low ASR** | Incorrect `attention_weight`. | Consult Table 10 in the paper; Llama-2-7B typically requires a weight of 150. |

## References
1. AttnGCG: Enhancing Adversarial Attacks on Language Models with Attention Manipulation
2. Prime Intellect H100 Hardware Specifications
