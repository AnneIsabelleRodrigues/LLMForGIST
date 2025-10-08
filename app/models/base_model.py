from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

from app.config import HUGGINGFACE_TOKEN

def load_base_model(model_name="Qwen/Qwen1.5-0.5B", load_in_8bit=True): #Qwen/Qwen2.5-7B
    login(HUGGINGFACE_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=HUGGINGFACE_TOKEN,
        trust_remote_code=True
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model