# ReasoningQwen2.5
# Qwen 2.5 Reasoning Fine-tune 🧠

This project demonstrates how to fine-tune the **Qwen 2.5 (0.5B Instruct)** model to enhance its reasoning capabilities using **QLoRA** (Quantized Low-Rank Adaptation).

The model is trained on the **OpenThoughts-114k** dataset to improve Chain-of-Thought (CoT) generation and is evaluated on the **GSM8k** benchmark.

## 🚀 Key Features
* **Efficient Fine-tuning:** Uses 4-bit quantization (BitsAndBytes) and LoRA to train on consumer-grade GPUs.
* **Modular Codebase:** Logic is separated into reusable modules (`src/`) for better maintainability.
* **Reasoning Focus:** Specifically prompts the model to generate `<thinking>` steps before providing the solution.
* **Evaluation Pipeline:** Includes scripts to merge adapters and evaluate performance using `lm-evaluation-harness`.

## 📂 Project Structure
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
