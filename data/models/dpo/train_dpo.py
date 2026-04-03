"""Auto-generated DPO training script. Run with: uv run python data/models/dpo/train_dpo.py"""
import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer
import torch

# Load config
config = json.loads(Path("data\models\dpo\dpo_config.json").read_text())

# Load data
pairs = []
with open(config["data_path"]) as f:
    for line in f:
        pairs.append(json.loads(line))

dataset = Dataset.from_list([{
    "prompt": p["prompt"],
    "chosen": p["chosen"],
    "rejected": p["rejected"],
} for p in pairs])

print(f"Training on {len(dataset)} preference pairs")

# Load model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    config["base_model"],
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
tokenizer.pad_token = tokenizer.eos_token

# LoRA config
lora_config = LoraConfig(
    r=config["lora_r"],
    lora_alpha=config["lora_alpha"],
    lora_dropout=config["lora_dropout"],
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# DPO training config
training_args = DPOConfig(
    output_dir=config["output_dir"],
    num_train_epochs=config["num_train_epochs"],
    per_device_train_batch_size=config["per_device_train_batch_size"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    learning_rate=config["learning_rate"],
    beta=config["beta"],
    max_length=config["max_length"],
    max_prompt_length=config["max_prompt_length"],
    fp16=config["fp16"],
    logging_steps=10,
    save_steps=50,
    remove_unused_columns=False,
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
)

print("Starting DPO training...")
trainer.train()
trainer.save_model(config["output_dir"] + "/final")
print(f"Model saved to {config['output_dir']}/final")
