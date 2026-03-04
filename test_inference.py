
import torch
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
from threading import Thread
import sys
import json
import base64

model_name = "unsloth/Qwen2.5-3B-Instruct" # Using a smaller model for faster testing
lora_path = None
messages = [{"role": "user", "content": "Tell me a short joke."}]

try:
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = lora_path if lora_path else model_name,
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded.")

    # Use chat template if available, otherwise fallback to simple join
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")
        print("Inputs prepared with chat template.")
    except Exception as e:
        print(f"Warning: apply_chat_template failed: {e}. Using fallback formatting.")
        prompt = ""
        for msg in messages:
            role = msg['role'].upper()
            content = msg['content']
            prompt += f"### {role}:\n{content}\n\n"
        prompt += "### ASSISTANT:\n"
        inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt = True, skip_special_tokens = True)

    generation_kwargs = dict(
        input_ids = inputs if torch.is_tensor(inputs) else inputs["input_ids"],
        streamer = streamer,
        max_new_tokens = 64,
        use_cache = True
    )

    print("Starting generation...")
    thread = Thread(target = model.generate, kwargs = generation_kwargs)
    thread.start()

    for new_text in streamer:
        print(f"TOKEN_OUTPUT: {new_text}", flush=True)

    print("Generation complete.")

except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
    sys.exit(1)
