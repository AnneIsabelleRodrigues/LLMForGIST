from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

from app.config import HUGGINGFACE_TOKEN

def load_base_model(model_name="Qwen/Qwen3-4B-Instruct-2507", load_in_4bit: bool = False): #Qwen/Qwen2.5-7B

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

    return tokenizer, model