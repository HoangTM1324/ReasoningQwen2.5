# ReasoningQwen2.5
# Qwen 2.5 Reasoning Fine-tune 

This project demonstrates how to fine-tune the **Qwen 2.5 (0.5B Instruct)** model to enhance its reasoning capabilities using **QLoRA** (Quantized Low-Rank Adaptation).

The model is trained on the **OpenThoughts-114k** dataset to improve Chain-of-Thought (CoT) generation and is evaluated on the **GSM8k** benchmark.

## Key Features
* **Efficient Fine-tuning:** Uses 4-bit quantization (BitsAndBytes) and LoRA to train on consumer-grade GPUs.
* **Modular Codebase:** Logic is separated into reusable modules (`src/`) for better maintainability.
* **Reasoning Focus:** Specifically prompts the model to generate `<thinking>` steps before providing the solution.
* **Evaluation Pipeline:** Includes scripts to merge adapters and evaluate performance using `lm-evaluation-harness`.

## Project Structure
```text
.
├── notebooks/
│   ├── 01_finetune_qwen.ipynb      # Main training notebook
│   └── 02_evaluate_model.ipynb     # Merging and benchmarking
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── data_utils.py               # Dataset processing & formatting
│   ├── model_utils.py              # Model loading & merging logic
│   └── trainer.py                  # LoRA config & Trainer setup
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```
## Installation
### Clone the repository:
```text
git clone [[https://github.com/your-username/qwen-reasoning-finetune.git](https://github.com/your-username/qwen-reasoning-finetune.git)
cd qwen-reasoning-finetune](https://github.com/HoangTM1324/ReasoningQwen2.5.git)
```
### Install dependencies:
```text
pip install -r requirements.txt
```
### (Optional) Install Flash Attention 2 for faster training (requires compatible GPU):
```text
pip install flash-attn --no-build-isolation
```

## Usage
### Fine-tuning
Run the training notebook or script. This will download the Qwen 2.5 model and the OpenThoughts dataset, then start fine-tuning.

Input: Qwen/Qwen2.5-0.5B-Instruct
Dataset: open-thoughts/OpenThoughts-114k

### Evaluation
To evaluate the model, we first merge the LoRA adapters into the base model, then run the GSM8k benchmark.
```text
from src import merge_lora_and_save

# Merge and save the final model
merge_lora_and_save(
    base_model_id="Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path="./output/checkpoint-final",
    save_path="./merged-qwen-model"
)
```
Run evaluation using lm_eval:
```
lm_eval --model hf --model_args pretrained=./merged-qwen-model --tasks gsm8k --batch_size 4
```
## Acknowledgments
* Base Model: Qwen 2.5
* Dataset: OpenThoughts
