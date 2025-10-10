from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


def apply_lora(model, r=8, alpha=32, dropout=0.05):
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    return model

def fine_tune(model, tokenizer, dataset, output_dir="./results"):

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=False, #mudar
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation']
    )

    trainer.train()
    model.save_pretrained(f"{output_dir}/lora_model")
    tokenizer.save_pretrained(f"{output_dir}/lora_model")