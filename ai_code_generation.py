import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import os

# Configure environment
os.environ['TRANSFORMERS_CACHE'] = '/path/to/transformers/cache'

async def load_model(model_name: str) -> AutoModelForCausalLM:
    """
    Asynchronously loads a transformer model for code generation tasks.

    :param model_name: The name of the model to load.
    :return: Instantiated model for causal language modeling.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model

async def load_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Asynchronously loads a tokenizer compatible with the given model.

    :param model_name: The name of the model to load the tokenizer for.
    :return: Tokenizer instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

async def generate_code(model_name: str, prompt: str) -> str:
    """
    Generates code based on a given prompt using a pre-trained transformer model.

    :param model_name: Name of the pre-trained model.
    :param prompt: Input prompt for code generation.
    :return: Generated code as a string.
    """
    model = await load_model(model_name)
    tokenizer = await load_tokenizer(model_name)

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    output = model.generate(input_ids, max_length=200, num_return_sequences=1)

    return tokenizer.decode(output[0], skip_special_tokens=True)

async def main():
    model_name = "codellama/code-llama-7b"
    prompt = "def fast_sort(arr):"
    try:
        generated_code = await generate_code(model_name, prompt)
        print("Generated Code:\n", generated_code)
    except Exception as e:
        print("Error during code generation:", e)

# To test the code
asyncio.run(main())