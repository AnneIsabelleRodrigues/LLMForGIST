# app/models/fine_tune.py
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model


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

    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=256
        )

    tokenized_ds = dataset.map(tokenize_fn, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds
    )

    trainer.train()
    model.save_pretrained(f"{output_dir}/lora_model")
