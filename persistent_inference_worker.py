
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from peft import PeftModel
from threading import Thread
import sys
import json
import base64
import traceback

def log(msg):
    print(f"WORKER_LOG: {msg}", flush=True)

def main():
    log("Starting persistent inference worker...")

    current_model_id = None
    current_lora_path = None
    model = None
    tokenizer = None

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            command = json.loads(line)
            action = command.get('action')

            if action == 'load':
                model_id = command.get('model_id')
                lora_path = command.get('lora_path')

                log(f"Loading model: {model_id} (LoRA: {lora_path})")

                # Cleanup if already loaded
                if model:
                    del model
                    del tokenizer
                    torch.cuda.empty_cache()

                tokenizer = AutoTokenizer.from_pretrained(model_id)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    attn_implementation="sdpa",
                    trust_remote_code=True
                )

                if lora_path:
                    model = PeftModel.from_pretrained(model, lora_path)
                    log("LoRA adapters loaded.")

                model.eval()
                current_model_id = model_id
                current_lora_path = lora_path
                log("LOAD_COMPLETE")

            elif action == 'chat':
                if not model:
                    log("ERROR: No model loaded")
                    continue

                messages = command.get('messages')
                max_new_tokens = command.get('max_new_tokens', 256)

                try:
                    inputs = tokenizer.apply_chat_template(
                        messages,
                        tokenize = True,
                        add_generation_prompt = True,
                        return_tensors = "pt",
                    ).to("cuda")
                except:
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
                    max_new_tokens = max_new_tokens,
                    use_cache = True,
                    do_sample = True,
                    temperature = 0.7,
                    top_p = 0.9,
                )

                thread = Thread(target = model.generate, kwargs = generation_kwargs)
                thread.start()

                print("CHAT_START", flush=True)
                for new_text in streamer:
                    # Escape newlines for single-line protocol
                    safe_text = new_text.replace('\n', '\\n').replace('\r', '\\r')
                    print(f"TOKEN: {safe_text}", flush=True)
                print("CHAT_END", flush=True)

            elif action == 'status':
                status = {
                    'loaded': model is not None,
                    'model_id': current_model_id,
                    'lora_path': current_lora_path
                }
                print(f"STATUS_JSON: {json.dumps(status)}", flush=True)

        except Exception as e:
            log(f"CRITICAL_ERROR: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
