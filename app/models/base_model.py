from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login
import torch

from ..config import HUGGINGFACE_TOKEN


def load_base_model(model_name="mistralai/Ministral-8B-Instruct-2410",
                    load_in_4bit: bool = True):

    login(HUGGINGFACE_TOKEN)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HUGGINGFACE_TOKEN,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
        token=HUGGINGFACE_TOKEN
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    return tokenizer, model


def load_gemma_model(model_name="google/medgemma-4b-it", load_in_4bit: bool = True):
    login(HUGGINGFACE_TOKEN)

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HUGGINGFACE_TOKEN,
    )

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16"
        )

    return processor, model


def load_llama_model(model_name="meta-llama/Llama-3.1-8B", load_in_4bit: bool = True):
    login(HUGGINGFACE_TOKEN)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HUGGINGFACE_TOKEN,
    )

    return tokenizer, model