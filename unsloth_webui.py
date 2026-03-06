#!/usr/bin/env python3
import eventlet
eventlet.monkey_patch()

import os
import json
import threading
import subprocess
import sys
import time
import shutil
import requests
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, session
from flask_socketio import SocketIO, emit
import zipfile
import tempfile
from werkzeug.utils import secure_filename
import hashlib
from functools import wraps

app = Flask(__name__)
app.config['SECRET_KEY'] = 'unsloth-secret-key-change-this'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024 * 1024  # 500GB max
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=100*1024*1024, logger=True, engineio_logger=True)

# Configurare
CONFIG_FILE = 'unsloth_config.json'
WORK_DIR = None
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
OUTPUT_FOLDER = 'outputs'
MERGED_FOLDER = 'merged_models'
CUSTOM_MODELS_FILE = 'custom_models.json'
ALLOWED_EXTENSIONS = {'json', 'jsonl', 'csv', 'txt', 'parquet', 'zip'}

# Dicționar pentru a stoca thread-urile active
active_threads = {}
# Mutex for GPU access to prevent OOM
gpu_lock = threading.Lock()

class PersistentInferenceManager:
    def __init__(self):
        self.process = None
        self.current_model = None
        self.current_lora = None
        self.status = "idle" # idle, loading, ready
        self.lock = threading.Lock()

    def start_worker(self, model_id, lora_path=None):
        with self.lock:
            if self.process:
                self.stop_worker()

            self.status = "loading"
            self.current_model = model_id
            self.current_lora = lora_path

            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'

            script_path = os.path.join(os.path.dirname(__file__), 'persistent_inference_worker.py')
            self.process = subprocess.Popen(
                [sys.executable, script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                env=env
            )

            # Start a thread to read logs/status
            threading.Thread(target=self._monitor_output, daemon=True).start()

            # Send load command
            cmd = json.dumps({'action': 'load', 'model_id': model_id, 'lora_path': lora_path})
            self.process.stdin.write(cmd + '\n')
            self.process.stdin.flush()

    def stop_worker(self):
        with self.lock:
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except:
                    self.process.kill()
                self.process = None
            self.current_model = None
            self.current_lora = None
            self.status = "idle"

    def chat(self, messages, max_new_tokens=256):
        if not self.process or self.status != "ready":
            return False

        cmd = json.dumps({'action': 'chat', 'messages': messages, 'max_new_tokens': max_new_tokens})
        self.process.stdin.write(cmd + '\n')
        self.process.stdin.flush()
        return True

    def _monitor_output(self):
        while self.process:
            line = self.process.stdout.readline()
            if not line:
                break

            line = line.strip()
            if line.startswith("WORKER_LOG: "):
                msg = line.replace("WORKER_LOG: ", "")
                print(f"🤖 [Worker] {msg}")
                if "LOAD_COMPLETE" in msg:
                    self.status = "ready"
                    socketio.emit('vram_status', {'status': 'ready', 'model': self.current_model})
                elif "CRITICAL_ERROR" in msg:
                    self.status = "error"
                    socketio.emit('vram_status', {'status': 'error', 'message': msg})

            elif line == "CHAT_START":
                socketio.emit('inference_output', {'data': "💬 (Persistent Model) Generând răspuns...\n"})

            elif line.startswith("TOKEN: "):
                token = line.replace("TOKEN: ", "").replace('\\n', '\n').replace('\\r', '\r')
                socketio.emit('inference_token', {'token': token})

            elif line == "CHAT_END":
                socketio.emit('inference_complete', {'success': True, 'persistent': True})

        print("🤖 [Worker] Process exited.")
        self.status = "idle"
        socketio.emit('vram_status', {'status': 'idle'})

vram_manager = PersistentInferenceManager()
# Buffer pentru ultimele log-uri de antrenare
training_logs_buffer = []
MAX_LOG_BUFFER = 500

# Decorator pentru logging API
def log_request(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            print(f"📡 API Request: {request.method} {request.path}")
            # Use repr to safely print non-ASCII headers
            print(f"📦 Headers: {repr(dict(request.headers))}")
            if request.method == 'POST':
                try:
                    # Only try to log JSON if it's actually JSON, not multipart/form-data
                    if request.is_json:
                        print(f"📦 Data: {repr(request.get_json(silent=True))}")
                    else:
                        print(f"📦 Data: Non-JSON body (likely file upload)")
                except:
                    print(f"📦 Data: [Error reading JSON data]")
        except Exception as e:
            print(f"⚠️ Error in log_request prefix: {e}")

        try:
            response = f(*args, **kwargs)
            print(f"✅ API Response: {request.path} - Status OK")
            return response
        except Exception as e:
            print(f"❌ API Error: {request.path} - {repr(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500
    return decorated_function

def load_config():
    global WORK_DIR
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            WORK_DIR = config.get('working_dir')
            if WORK_DIR and os.path.exists(WORK_DIR):
                print(f"✅ Loaded working directory: {WORK_DIR}")
                return WORK_DIR
            else:
                print(f"⚠️ Working directory not found: {WORK_DIR}")
    except FileNotFoundError:
        print("📝 No config file found")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
    return None

def save_config(wd):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump({'working_dir': wd}, f)
        print(f"✅ Saved working directory: {wd}")
    except Exception as e:
        print(f"❌ Error saving config: {e}")

def get_work_path(subfolder):
    """Returnează calea completă către un subfolder în WORK_DIR"""
    if not WORK_DIR:
        raise ValueError("WORK_DIR not set")
    path = os.path.join(WORK_DIR, subfolder)
    os.makedirs(path, exist_ok=True)
    return path

def load_custom_models():
    """Încarcă lista de modele custom"""
    try:
        if not WORK_DIR:
            return {}
        models_file = os.path.join(WORK_DIR, CUSTOM_MODELS_FILE)
        if os.path.exists(models_file):
            with open(models_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"❌ Error loading custom models: {e}")
    return {}

def save_custom_models(models):
    """Salvează lista de modele custom"""
    try:
        if not WORK_DIR:
            return
        models_file = os.path.join(WORK_DIR, CUSTOM_MODELS_FILE)
        with open(models_file, 'w') as f:
            json.dump(models, f, indent=2)
    except Exception as e:
        print(f"❌ Error saving custom models: {e}")

# Modele disponibile
AVAILABLE_MODELS = {
    "Qwen2.5-3B-Instruct": "unsloth/Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct": "unsloth/Qwen2.5-7B-Instruct",
    "Qwen2.5-14B-Instruct": "unsloth/Qwen2.5-14B-Instruct",
    "Qwen2.5-32B-Instruct": "unsloth/Qwen2.5-32B-Instruct",
    "Qwen2.5-72B-Instruct": "unsloth/Qwen2.5-72B-Instruct",
    "Llama-3.2-3B-Instruct": "unsloth/Llama-3.2-3B-Instruct",
    "Llama-3.2-1B-Instruct": "unsloth/Llama-3.2-1B-Instruct",
    "Llama-3.1-8B-Instruct": "unsloth/Llama-3.1-8B-Instruct",
    "Llama-3.1-70B-Instruct": "unsloth/Llama-3.1-70B-Instruct",
    "Mistral-7B-Instruct-v0.3": "unsloth/mistral-7b-instruct-v0.3",
    "Mixtral-8x7B-Instruct": "unsloth/mixtral-8x7b-instruct",
    "Gemma-2-9B-it": "unsloth/gemma-2-9b-it",
    "Gemma-2-27B-it": "unsloth/gemma-2-27b-it",
    "Phi-3.5-mini-instruct": "unsloth/Phi-3.5-mini-instruct",
    "Phi-3.5-medium-instruct": "unsloth/Phi-3.5-medium-instruct",
    "DeepSeek-R1-Distill-Qwen-7B": "unsloth/DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Qwen-32B": "unsloth/DeepSeek-R1-Distill-Qwen-32B",
}

class TrainingThread(threading.Thread):
    def __init__(self, sid, config):
        threading.Thread.__init__(self)
        self.sid = sid
        self.config = config
        self.daemon = True
        self.process = None
        self.session_id = str(int(time.time()))
        
    def run(self):
        print(f"🧵 Training thread {self.session_id} waiting for GPU lock...")
        with gpu_lock:
            print(f"🧵 Training thread {self.session_id} acquired GPU lock. SID: {self.sid}")
            self._do_run()

    def _do_run(self):
        try:
            # Switch to non-blocking and proper log capture
            import sys

            # Verifică dacă e model custom
            model_name = self.config['model_name']
            if model_name.startswith('custom:'):
                model_path = model_name.replace('custom:', '')
                # Relax existence check to support HuggingFace repo IDs (which contain '/')
                if not os.path.exists(model_path) and "/" not in model_path:
                    socketio.emit('training_output', {
                        'data': f"❌ Model path not found and does not look like a HuggingFace repo: {model_path}\n"
                    })
                    socketio.emit('training_complete', {
                        'success': False,
                        'error': 'Model path not found or invalid HuggingFace ID'
                    })
                    return
                model_name = model_path
            
            # Crează scriptul de antrenare
            script_path = os.path.join(WORK_DIR, f'train_script_{self.session_id}.py')
            with open(script_path, 'w') as f:
                f.write(self.generate_training_script(model_name))
            
            socketio.emit('training_output', {
                'data': "🚀 Starting training with configuration:\n"
            })
            socketio.emit('training_output', {
                'data': json.dumps(self.config, indent=2) + "\n"
            })
            
            # Rulează antrenarea
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            env['CUDA_VISIBLE_DEVICES'] = '0'
            env['SESSION_ID'] = self.session_id
            
            print(f"🚀 Spawning training process: {sys.executable} {script_path}")
            self.process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                env=env,
                cwd=WORK_DIR
            )
            
            while True:
                line = self.process.stdout.readline()
                if not line and self.process.poll() is not None:
                    break

                if not line:
                    continue

                # Store in buffer
                training_logs_buffer.append(line)
                if len(training_logs_buffer) > MAX_LOG_BUFFER:
                    training_logs_buffer.pop(0)

                # Robust parsing for PROGRESS_JSON, even if embedded in tqdm updates (\r)
                if 'PROGRESS_JSON: ' in line:
                    # Split by \r to get the latest part if multiple tqdm updates are on one line
                    parts = line.replace('\r', '\n').split('\n')
                    for part in parts:
                        if 'PROGRESS_JSON: ' in part:
                            try:
                                json_str = part.split('PROGRESS_JSON: ')[1].strip()
                                progress_data = json.loads(json_str)
                                socketio.emit('training_progress', progress_data)

                                # Send any prefix to the console
                                prefix = part.split('PROGRESS_JSON: ')[0].strip()
                                if prefix:
                                    socketio.emit('training_output', {'data': prefix + '\n'})
                            except Exception as e:
                                print(f"Error parsing progress JSON: {e}")
                        elif part.strip():
                            socketio.emit('training_output', {'data': part + '\n'})
                else:
                    # Broadcast to all clients
                    socketio.emit('training_output', {'data': line})
            
            self.process.wait()
            print(f"🏁 Training process finished with return code {self.process.returncode}")
            
            # Curăță scriptul temporar
            try:
                os.remove(script_path)
            except:
                pass
            
            if self.process.returncode == 0:
                socketio.emit('training_complete', {
                    'success': True,
                    'output_dir': self.config.get('output_dir', 'outputs/final_model')
                })
            else:
                socketio.emit('training_complete', {
                    'success': False,
                    'error': f'Training failed with return code {self.process.returncode}'
                })
                
        except Exception as e:
            print(f"❌ Error in training thread: {e}")
            socketio.emit('training_complete', {
                'success': False,
                'error': str(e)
            })
    
    def generate_training_script(self, model_name):
        model_name_repr = repr(model_name)
        dataset_path_repr = repr(self.config['dataset_path'])
        output_dir_repr = repr(self.config['output_dir'])

        return f'''import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer
import json
import traceback

class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            progress_data = {{
                "step": state.global_step,
                "max_steps": state.max_steps,
                "epoch": round(state.epoch, 2) if state.epoch is not None else None,
                "loss": logs.get("loss"),
                "learning_rate": logs.get("learning_rate"),
            }}
            if state.max_steps > 0:
                progress_data["progress"] = round((state.global_step / state.max_steps) * 100, 2)
            js_str = json.dumps(progress_data)
            print(f"\\nPROGRESS_JSON: {{js_str}}", flush=True)

try:
    # Configurare
    model_name = {model_name_repr}
    dataset_path = {dataset_path_repr}
    output_dir = {output_dir_repr}
    max_seq_length = {self.config.get('max_seq_length', 2048)}
    load_in_4bit = {self.config.get('load_in_4bit', True)}
    batch_size = {self.config.get('batch_size', 2)}
    gradient_accumulation = {self.config.get('gradient_accumulation', 4)}
    learning_rate = {self.config.get('learning_rate', 2e-4)}
    num_epochs = {self.config.get('num_epochs', 3)}
    max_steps = {self.config.get('max_steps', 0)}
    warmup_steps = {self.config.get('warmup_steps', 5)}
    save_steps = {self.config.get('save_steps', 50)}
    logging_steps = {self.config.get('logging_steps', 1)}
    lora_r = {self.config.get('lora_r', 16)}
    lora_alpha = {self.config.get('lora_alpha', 16)}
    lora_dropout = {self.config.get('lora_dropout', 0)}
    use_gradient_checkpointing = {self.config.get('use_gradient_checkpointing', True)}
    optim = "{self.config.get('optim', 'adamw_8bit')}"
    packing = {self.config.get('packing', False)}

    print("🚀 Loading model...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            device_map="auto",
        )
        print(f"✅ Model loaded: {{model_name}}")
    except Exception as e:
        print(f"❌ Error loading model: {{e}}")
        traceback.print_exc()
        raise

    print("🔧 Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth" if use_gradient_checkpointing else False,
        random_state=42,
    )

    print("📚 Loading dataset...")
    try:
        if dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
            dataset = load_dataset('json', data_files=dataset_path, split='train')
        elif dataset_path.endswith('.csv'):
            dataset = load_dataset('csv', data_files=dataset_path, split='train')
        elif dataset_path.endswith('.parquet'):
            dataset = load_dataset('parquet', data_files=dataset_path, split='train')
        else:
            raise ValueError(f"Unsupported dataset format: {{dataset_path}}")
        
        print(f"✅ Dataset loaded: {{len(dataset)}} examples")
    except Exception as e:
        print(f"❌ Error loading dataset: {{e}}")
        traceback.print_exc()
        raise

    EOS_TOKEN = tokenizer.eos_token
    def format_prompt(example):
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output = example.get('output', '')
        
        if input_text:
            text = f"### Instruction:\\n{{instruction}}\\n\\n### Input:\\n{{input_text}}\\n\\n### Response:\\n{{output}}{{EOS_TOKEN}}"
        else:
            text = f"### Instruction:\\n{{instruction}}\\n\\n### Response:\\n{{output}}{{EOS_TOKEN}}"

        # Explicitly truncate text slightly below max_seq_length to avoid Dynamo batch size mismatches
        tokenized = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_seq_length - 4)
        text = tokenizer.decode(tokenized)
        
        return {{"text": text}}

    dataset = dataset.map(format_prompt)
    print("✅ Dataset formatted")

    print("🎯 Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=4,
        packing=packing,
        callbacks=[ProgressCallback()],
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs if max_steps == 0 else 1,
            max_steps=max_steps if max_steps > 0 else -1,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=logging_steps,
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=2,
            remove_unused_columns=False,
            report_to="none",
            optim=optim,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
        ),
    )

    print("🏃 Training in progress...")
    trainer.train()

    print("💾 Saving LoRA adapters...")
    adapter_dir = os.path.join(output_dir, "lora_adapter")
    model.save_pretrained(adapter_dir)
    print(f"✅ LoRA adapters saved to {{adapter_dir}}")

    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump({{
            "base_model": model_name,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "dataset": dataset_path,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }}, f, indent=2)

    print("✅ Training complete! LoRA adapters are ready for merging.")
    
except Exception as e:
    print(f"❌ Fatal error: {{e}}")
    traceback.print_exc()
    raise
'''

class InferenceThread(threading.Thread):
    def __init__(self, sid, model_name, lora_path, messages, max_new_tokens=256):
        threading.Thread.__init__(self)
        self.sid = sid
        self.model_name = model_name
        self.lora_path = lora_path
        self.messages = messages # Expects a list of {"role": "user/assistant", "content": "..."}
        self.max_new_tokens = max_new_tokens
        self.daemon = True
        self.session_id = f"inf_{int(time.time())}"

    def run(self):
        print(f"🧵 Inference thread {self.session_id} waiting for GPU lock...")
        with gpu_lock:
            print(f"🧵 Inference thread {self.session_id} acquired GPU lock. SID: {self.sid}")
            try:
                socketio.emit('inference_output', {'data': "🚀 Loading model for inference...\n"})

                # We use a subprocess for inference too to keep the main process stable and separate VRAM
                script_path = os.path.join(WORK_DIR, f'inference_script_{self.session_id}.py')
                with open(script_path, 'w') as f:
                    f.write(self.generate_inference_script())

                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'

                process = subprocess.Popen(
                    [sys.executable, script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    universal_newlines=True,
                    env=env,
                    cwd=WORK_DIR
                )

                for line in process.stdout:
                    if 'TOKEN_OUTPUT: ' in line:
                        token = line.split('TOKEN_OUTPUT: ')[1].rstrip('\n')
                        socketio.emit('inference_token', {'token': token})
                    else:
                        socketio.emit('inference_output', {'data': line})

                process.wait()
                os.remove(script_path)
                socketio.emit('inference_complete', {'success': process.returncode == 0})

            except Exception as e:
                print(f"❌ Inference error: {e}")
                socketio.emit('inference_output', {'data': f"\n❌ Error: {str(e)}\n"})
                socketio.emit('inference_complete', {'success': False, 'error': str(e)})

    def generate_inference_script(self):
        # Using base64 to safely pass messages to the subprocess
        import json
        import base64
        messages_json = json.dumps(self.messages)
        messages_b64 = base64.b64encode(messages_json.encode('utf-8')).decode('utf-8')

        # Use repr() for path variables to correctly handle None vs string
        model_name_repr = repr(self.model_name)
        lora_path_repr = repr(self.lora_path)

        return f'''
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

model_name = {model_name_repr}
lora_path = {lora_path_repr}
messages_b64 = "{messages_b64}"

try:
    print(f"🔍 Loading model: {{model_name}}", flush=True)
    messages = json.loads(base64.b64decode(messages_b64).decode('utf-8'))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configuration for high-performance 4-bit inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Use SDPA (Scaled Dot Product Attention) for high performance
    # This automatically leverages FlashAttention-2 if compatible hardware/drivers are present
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        trust_remote_code=True
    )

    if lora_path:
        print(f"📦 Loading LoRA adapters: {{lora_path}}", flush=True)
        model = PeftModel.from_pretrained(model, lora_path)
        print("✅ LoRA adapters merged into base model.", flush=True)

    model.eval()
    print("✅ Model loaded with SDPA/FlashAttention acceleration.", flush=True)

    # Use chat template if available, otherwise fallback to simple join
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")
        print("📝 Chat template applied.", flush=True)
    except Exception as e:
        print(f"⚠️ Warning: apply_chat_template failed: {{e}}. Using fallback formatting.", flush=True)
        prompt = ""
        for msg in messages:
            role = msg['role'].upper()
            content = msg['content']
            prompt += f"### {{role}}:\\n{{content}}\\n\\n"
        prompt += "### ASSISTANT:\\n"
        inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt = True, skip_special_tokens = True)

    generation_kwargs = dict(
        input_ids = inputs if torch.is_tensor(inputs) else inputs["input_ids"],
        streamer = streamer,
        max_new_tokens = {self.max_new_tokens},
        use_cache = True,
        do_sample = True,
        temperature = 0.7,
        top_p = 0.9,
    )

    print("🚀 Starting generation...", flush=True)
    thread = Thread(target = model.generate, kwargs = generation_kwargs)
    thread.start()

    for new_text in streamer:
        print(f"TOKEN_OUTPUT: {{new_text}}", flush=True)

    print("\\n✅ Generation complete.", flush=True)

except Exception as e:
    print(f"❌ Subprocess error: {{e}}", flush=True)
    traceback.print_exc()
    sys.exit(1)
'''

class MergeThread(threading.Thread):
    def __init__(self, sid, base_model_path, lora_path, output_path, merge_method, quantization="none"):
        threading.Thread.__init__(self)
        self.sid = sid
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.output_path = output_path
        self.merge_method = merge_method
        self.quantization = quantization
        self.daemon = True
        self.process = None
        self.session_id = str(int(time.time()))
        
    def run(self):
        print(f"🧵 Merge thread {self.session_id} waiting for GPU lock...")
        with gpu_lock:
            print(f"🧵 Merge thread {self.session_id} acquired GPU lock. SID: {self.sid}")
            try:
                script_path = os.path.join(WORK_DIR, f'merge_script_{self.session_id}.py')
                with open(script_path, 'w') as f:
                    f.write(self.generate_merge_script())
            
                socketio.emit('merge_output', {
                    'data': "🔄 Starting LoRA merge process...\n"
                })

                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'

                print(f"🚀 Spawning merge process: {sys.executable} {script_path}")
                self.process = subprocess.Popen(
                    [sys.executable, script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    universal_newlines=True,
                    env=env,
                    cwd=WORK_DIR
                )

                for line in self.process.stdout:
                    # Echo subprocess output to server console for debugging
                    print(f"[Merge {self.session_id}] {line.strip()}")
                    # Broadcast to all clients
                    socketio.emit('merge_output', {'data': line})

                self.process.wait()
                print(f"🏁 Merge process finished with return code {self.process.returncode}")

                # Curăță scriptul temporar
                try:
                    os.remove(script_path)
                except:
                    pass

                if self.process.returncode == 0:
                    socketio.emit('merge_complete', {
                        'success': True,
                        'output_path': self.output_path
                    })
                else:
                    socketio.emit('merge_complete', {
                        'success': False,
                        'error': f'Merge failed with return code {self.process.returncode}'
                    })

            except Exception as e:
                print(f"❌ Error in merge thread: {e}")
                socketio.emit('merge_complete', {
                    'success': False,
                    'error': str(e)
                })
    
    def generate_merge_script(self):
        # Use repr() to safely pass paths
        base_model_repr = repr(self.base_model_path)
        lora_path_repr = repr(self.lora_path)
        output_path_repr = repr(self.output_path)

        return f'''import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from unsloth import FastLanguageModel
import shutil
import json
import traceback
import time
import sys

# Ensure stdout is flushed
def log(msg):
    print(msg, flush=True)

try:
    base_model = {base_model_repr}
    lora_path = {lora_path_repr}
    output_path = {output_path_repr}
    merge_method = "{self.merge_method}"
    quantization = "{self.quantization}"

    log(f"📂 Base model: {{base_model}}")
    log(f"📂 LoRA adapters: {{lora_path}}")
    log(f"📂 Output: {{output_path}}")

    log("🚀 Loading model and LoRA adapters...")
    try:
        # Use unsloth's optimized loader
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=lora_path if lora_path else base_model,
            max_seq_length=4096,
            load_in_4bit=(merge_method == "merged_4bit"),
            device_map="auto",
        )
        log("✅ Model loaded")
        
    except Exception as e:
        log(f"❌ Error loading model: {{e}}")
        traceback.print_exc()
        sys.exit(1)

    try:
        os.makedirs(output_path, exist_ok=True)
        
        if quantization != "none":
            log(f"📦 Starting GGUF export ({{quantization}})...")
            log("⏳ This process involves merging and then quantizing. It may take several minutes.")
            # unsloth will install llama.cpp if not present, which might explain the "system package" message
            model.save_pretrained_gguf(output_path, tokenizer, quantization_method = quantization)
            log("✅ GGUF export finished successfully")

            # Check if GGUF files were actually created
            files = os.listdir(output_path)
            gguf_files = [f for f in files if f.endswith(".gguf")]
            if gguf_files:
                log(f"📄 Created GGUF files: {{gguf_files}}")
            else:
                log("⚠️ Warning: No .gguf files found in output directory!")
        else:
            log(f"🔄 Starting LoRA merge ({{merge_method}})...")
            model.save_pretrained_merged(output_path, tokenizer, save_method = merge_method.replace("merged_", ""))
            log("✅ HF merge finished successfully")
        
        if os.path.exists(os.path.join(lora_path, "training_config.json")):
            shutil.copy(
                os.path.join(lora_path, "training_config.json"),
                os.path.join(output_path, "training_config.json")
            )
        
        # Adaugă un fișier README
        with open(os.path.join(output_path, "README.txt"), "w") as f:
            f.write(f"""Merged Model Information
======================
Base model: {{base_model}}
LoRA source: {{lora_path}}
Merge method: {{merge_method}}
Date: {{time.strftime("%Y-%m-%d %H:%M:%S")}}

This model was created by merging LoRA adapters with the base model.
It can be used with any framework that supports HuggingFace models.
""")
        
        print(f"✅ Merged model saved to {{output_path}}")
        print("🎉 Merge complete! Model is ready for deployment.")
        
    except Exception as e:
        print(f"❌ Error during merge: {{e}}")
        traceback.print_exc()
        raise
        
except Exception as e:
    print(f"❌ Fatal error: {{e}}")
    traceback.print_exc()
    raise
'''

@socketio.on('connect')
def handle_connect():
    global WORK_DIR
    print(f"🔌 Client connected: {request.sid}")
    if not WORK_DIR:
        emit('need_config')
    else:
        # Send current config and sid
        emit('connected', {
            'working_dir': WORK_DIR,
            'sid': request.sid,
            'active_training': any(t.is_alive() for t in active_threads.values())
        })

@socketio.on('disconnect')
def handle_disconnect():
    print(f"🔌 Client disconnected: {request.sid}")

@socketio.on('set_working_dir')
def handle_set_working_dir(data):
    global WORK_DIR
    wd = data.get('working_dir')
    print(f"📁 Setting working directory to: {wd}")
    
    if wd and os.path.exists(wd):
        WORK_DIR = os.path.abspath(wd)
        save_config(WORK_DIR)
        for folder in [UPLOAD_FOLDER, MODELS_FOLDER, OUTPUT_FOLDER, MERGED_FOLDER]:
            os.makedirs(os.path.join(WORK_DIR, folder), exist_ok=True)
        emit('config_saved', {'working_dir': WORK_DIR})
    else:
        emit('config_error', {'message': 'Invalid directory'})

# API Routes
@app.route('/api/hf/search', methods=['GET'])
@log_request
def hf_search():
    query = request.args.get('q', '')
    if not query:
        return jsonify([])

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # Search for models with the query, filtered by library and task
        models = api.list_models(
            search=query,
            filter="text-generation",
            sort="downloads",
            direction=-1,
            limit=20
        )

        results = []
        for m in models:
            results.append({
                'id': m.modelId,
                'downloads': getattr(m, 'downloads', 0),
                'likes': getattr(m, 'likes', 0),
                'updated': getattr(m, 'lastModified', '')
            })
        return jsonify(results)
    except Exception as e:
        print(f"❌ HF Search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
@log_request
def get_models():
    custom_models = load_custom_models()
    all_models = AVAILABLE_MODELS.copy()
    for name, path in custom_models.items():
        all_models[f"✨ {name} (custom)"] = f"custom:{path}"
    return jsonify(all_models)

@app.route('/api/custom_models', methods=['GET'])
@log_request
def list_custom_models():
    custom_models = load_custom_models()
    return jsonify(custom_models)

@app.route('/api/add_custom_model', methods=['POST'])
@log_request
def add_custom_model():
    data = request.get_json(silent=True) or {}
    name = data.get('name')
    path_or_hf = data.get('path')
    
    if not name or not path_or_hf:
        return jsonify({'success': False, 'error': 'Name and path required'}), 400
    
    custom_models = load_custom_models()
    custom_models[name] = path_or_hf
    save_custom_models(custom_models)
    
    return jsonify({'success': True, 'models': custom_models})

@app.route('/api/remove_custom_model/<name>', methods=['GET'])
@log_request
def remove_custom_model(name):
    custom_models = load_custom_models()
    if name in custom_models:
        del custom_models[name]
        save_custom_models(custom_models)
    return jsonify({'success': True})

@app.route('/api/files', methods=['GET'])
@log_request
def list_files():
    if not WORK_DIR:
        return jsonify({'error': 'No working directory'}), 400
    
    try:
        upload_path = get_work_path(UPLOAD_FOLDER)
        files = []
        for f in os.listdir(upload_path):
            if os.path.isfile(os.path.join(upload_path, f)):
                size = os.path.getsize(os.path.join(upload_path, f))
                files.append({
                    'name': f,
                    'size': size,
                    'size_human': f"{size / 1024 / 1024:.2f} MB" if size > 1024*1024 else f"{size / 1024:.2f} KB"
                })
        return jsonify(files)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trained_models', methods=['GET'])
@log_request
def list_trained_models():
    if not WORK_DIR:
        return jsonify({'error': 'No working directory'}), 400
    
    try:
        output_path = get_work_path(OUTPUT_FOLDER)
        models = []
        
        if not os.path.exists(output_path):
            return jsonify([])
        
        for item in os.listdir(output_path):
            item_path = os.path.join(output_path, item)
            if os.path.isdir(item_path):
                has_lora = os.path.exists(os.path.join(item_path, "lora_adapter", "adapter_config.json"))
                has_config = os.path.exists(os.path.join(item_path, "training_config.json"))
                
                total_size = 0
                for root, dirs, files in os.walk(item_path):
                    for f in files:
                        fp = os.path.join(root, f)
                        total_size += os.path.getsize(fp)
                
                models.append({
                    'name': item,
                    'path': item_path,
                    'has_lora': has_lora,
                    'has_config': has_config,
                    'size': total_size,
                    'size_human': f"{total_size / 1024 / 1024:.2f} MB" if total_size > 1024*1024 else f"{total_size / 1024:.2f} KB",
                    'created': time.ctime(os.path.getctime(item_path))
                })
        
        return jsonify(models)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/merged_models', methods=['GET'])
@log_request
def list_merged_models():
    if not WORK_DIR:
        return jsonify({'error': 'No working directory'}), 400
    
    try:
        merged_path = get_work_path(MERGED_FOLDER)
        models = []
        
        if not os.path.exists(merged_path):
            return jsonify([])
        
        for item in os.listdir(merged_path):
            item_path = os.path.join(merged_path, item)
            if os.path.isdir(item_path):
                has_config = os.path.exists(os.path.join(item_path, "config.json"))
                
                total_size = 0
                for root, dirs, files in os.walk(item_path):
                    for f in files:
                        fp = os.path.join(root, f)
                        total_size += os.path.getsize(fp)
                
                models.append({
                    'name': item,
                    'path': item_path,
                    'has_config': has_config,
                    'size': total_size,
                    'size_human': f"{total_size / 1024 / 1024:.2f} MB" if total_size > 1024*1024 else f"{total_size / 1024:.2f} KB",
                    'created': time.ctime(os.path.getctime(item_path))
                })
        
        return jsonify(models)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST', 'OPTIONS'])
@log_request
def upload_file():
    if request.method == 'OPTIONS':
        return jsonify({'success': True}), 200
        
    if not WORK_DIR:
        return jsonify({'error': 'No working directory'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    if not filename:
        # Fallback for filenames that are entirely non-ASCII and get stripped by secure_filename
        filename = f"uploaded_file_{int(time.time())}.json"

    upload_path = get_work_path(UPLOAD_FOLDER)
    filepath = os.path.join(upload_path, filename)
    
    # Salvează fișierul
    file.save(filepath)
    print(f"📁 File uploaded: {repr(filename)}")
    
    # Dacă e zip, extrage
    if filename.endswith('.zip'):
        try:
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(upload_path)
            os.remove(filepath)
            print(f"📦 ZIP extracted: {filename}")
            return jsonify({'success': True, 'message': 'ZIP extracted', 'files': os.listdir(upload_path)})
        except Exception as e:
            return jsonify({'error': f'Failed to extract ZIP: {str(e)}'}), 500
    
    return jsonify({'success': True, 'filename': filename})

@app.route('/api/download_model/<model_name>', methods=['GET'])
@log_request
def download_model(model_name):
    if not WORK_DIR:
        return jsonify({'error': 'No working directory'}), 400
    
    # Verifică în ambele locații
    output_path = get_work_path(OUTPUT_FOLDER)
    merged_path = get_work_path(MERGED_FOLDER)
    
    model_path = None
    source_type = None
    
    for base_path, type_name in [(output_path, 'lora'), (merged_path, 'merged')]:
        path = os.path.join(base_path, model_name)
        if os.path.exists(path):
            model_path = path
            source_type = type_name
            break
    
    if not model_path:
        return jsonify({'error': 'Model not found'}), 404
    
    # Crează un fișier temporar zip
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, f"{model_name}.zip")
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, model_path)
                    zipf.write(file_path, arcname)
        
        return send_file(zip_path, as_attachment=True, download_name=f"{model_name}.zip")
    except Exception as e:
        return jsonify({'error': f'Failed to create zip: {str(e)}'}), 500
    finally:
        # Curăță după trimitere
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

@app.route('/api/vram/load', methods=['POST'])
@log_request
def vram_load():
    data = request.get_json(silent=True) or {}
    model_id = data.get('model_identifier')

    if model_id.startswith('merged:'):
        model_name = os.path.join(get_work_path(MERGED_FOLDER), model_id.replace('merged:', ''))
    elif model_id.startswith('custom:'):
        model_name = model_id.replace('custom:', '')
    else:
        model_name = model_id

    lora_name = data.get('lora_name')
    lora_path = None
    if lora_name and lora_name != "none":
        lora_path = os.path.join(get_work_path(OUTPUT_FOLDER), lora_name, "lora_adapter")
        if not os.path.exists(lora_path):
            lora_path = os.path.join(get_work_path(OUTPUT_FOLDER), lora_name)

    vram_manager.start_worker(model_name, lora_path)
    return jsonify({'success': True, 'status': 'loading'})

@app.route('/api/vram/unload', methods=['GET'])
@log_request
def vram_unload():
    vram_manager.stop_worker()
    return jsonify({'success': True, 'status': 'idle'})

@app.route('/api/vram/status', methods=['GET'])
@log_request
def vram_status_api():
    return jsonify({
        'status': vram_manager.status,
        'model': vram_manager.current_model,
        'lora': vram_manager.current_lora
    })

@app.route('/api/start_inference', methods=['POST'])
@log_request
def start_inference():
    if not WORK_DIR:
        return jsonify({'error': 'No working directory'}), 400

    data = request.get_json(silent=True) or {}
    messages = data.get('messages')

    if not messages:
        return jsonify({'error': 'No messages provided'}), 400

    # If vram worker is ready, use it
    if vram_manager.status == "ready":
        success = vram_manager.chat(messages)
        return jsonify({'success': success, 'persistent': True})

    # Fallback to subprocess (existing logic)
    model_id = data.get('model_identifier')
    if model_id.startswith('merged:'):
        model_name = os.path.join(get_work_path(MERGED_FOLDER), model_id.replace('merged:', ''))
    elif model_id.startswith('custom:'):
        model_name = model_id.replace('custom:', '')
    else:
        model_name = model_id

    lora_name = data.get('lora_name')
    socket_id = data.get('socket_id')
    lora_path = None
    if lora_name and lora_name != "none":
        lora_path = os.path.join(get_work_path(OUTPUT_FOLDER), lora_name, "lora_adapter")
        if not os.path.exists(lora_path):
            lora_path = os.path.join(get_work_path(OUTPUT_FOLDER), lora_name)

    thread = InferenceThread(socket_id, model_name, lora_path, messages)
    socketio.start_background_task(thread.run)
    active_threads[thread.session_id] = thread

    return jsonify({'success': True, 'session_id': thread.session_id, 'persistent': False})

@app.route('/api/start_training', methods=['POST'])
@log_request
def start_training():
    if not WORK_DIR:
        return jsonify({'error': 'No working directory'}), 400
    
    data = request.get_json(silent=True) or {}
    print(f"📦 Received training data: {json.dumps(data, indent=2)}")
    
    socket_id = data.get('socket_id')
    print(f"🔌 Socket ID from request: {socket_id}")
    
    if not socket_id:
        return jsonify({'error': 'No socket_id provided'}), 400
    
    timestamp = int(time.time())
    output_name = data.get('output_name', f'model_{timestamp}')
    output_dir = os.path.join(
        get_work_path(OUTPUT_FOLDER), 
        output_name
    )
    
    # Verifică dacă dataset-ul există
    dataset_path = os.path.join(get_work_path(UPLOAD_FOLDER), data['dataset_file'])
    if not os.path.exists(dataset_path):
        return jsonify({'error': f'Dataset file not found: {dataset_path}'}), 400
    
    config = {
        'model_name': data['model_name'],
        'dataset_file': data['dataset_file'],
        'output_name': output_name,
        'output_dir': output_dir,
        'dataset_path': dataset_path,
        'max_seq_length': data.get('max_seq_length', 2048),
        'batch_size': data.get('batch_size', 2),
        'learning_rate': data.get('learning_rate', 2e-4),
        'num_epochs': data.get('num_epochs', 3),
        'max_steps': data.get('max_steps', 0),
        'warmup_steps': data.get('warmup_steps', 5),
        'save_steps': data.get('save_steps', 50),
        'logging_steps': data.get('logging_steps', 1),
        'optim': data.get('optim', 'adamw_8bit'),
        'packing': data.get('packing', False),
        'gradient_accumulation': data.get('gradient_accumulation', 4),
        'lora_r': data.get('lora_r', 16),
        'lora_alpha': data.get('lora_alpha', 16),
        'load_in_4bit': data.get('load_in_4bit', True),
    }
    
    # Pornește thread-ul de antrenare
    thread = TrainingThread(socket_id, config)

    # Folosește background task din SocketIO pentru o mai bună integrare
    socketio.start_background_task(thread.run)
    active_threads[thread.session_id] = thread
    
    return jsonify({
        'success': True, 
        'message': 'Training started', 
        'session_id': thread.session_id,
        'output_dir': output_dir
    })

@app.route('/api/start_merge', methods=['POST'])
@log_request
def start_merge():
    if not WORK_DIR:
        return jsonify({'error': 'No working directory'}), 400
    
    data = request.get_json(silent=True) or {}
    print(f"📦 Received merge data: {json.dumps(data, indent=2)}")
    
    socket_id = data.get('socket_id')
    trained_model = data.get('trained_model')
    merge_method = data.get('merge_method', 'merged_16bit')
    quantization = data.get('quantization', 'none')
    
    print(f"🔌 Socket ID from request: {socket_id}")
    
    if not socket_id:
        return jsonify({'error': 'No socket_id provided'}), 400
    
    trained_path = os.path.join(get_work_path(OUTPUT_FOLDER), trained_model)
    lora_path = os.path.join(trained_path, "lora_adapter")
    
    if not os.path.exists(lora_path):
        return jsonify({'error': 'LoRA adapters not found in trained model'}), 400
    
    config_path = os.path.join(trained_path, "training_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            train_config = json.load(f)
            base_model = train_config.get('base_model')
    else:
        return jsonify({'error': 'Training config not found. Cannot determine base model.'}), 400
    
    timestamp = int(time.time())
    output_name = f"{trained_model}_merged_{timestamp}"
    output_path = os.path.join(get_work_path(MERGED_FOLDER), output_name)
    
    thread = MergeThread(socket_id, base_model, lora_path, output_path, merge_method, quantization)
    socketio.start_background_task(thread.run)
    active_threads[thread.session_id] = thread
    
    return jsonify({'success': True, 'message': 'Merge started', 'session_id': thread.session_id})

@app.route('/api/delete_file/<filename>', methods=['GET'])
@log_request
def delete_file(filename):
    if not WORK_DIR:
        return jsonify({'error': 'No working directory'}), 400
    
    upload_path = get_work_path(UPLOAD_FOLDER)
    filepath = os.path.join(upload_path, filename)
    
    if os.path.exists(filepath):
        os.remove(filepath)
        return jsonify({'success': True})
    
    return jsonify({'error': 'File not found'}), 404

@app.route('/api/delete_model/<model_name>', methods=['GET'])
@log_request
def delete_model(model_name):
    if not WORK_DIR:
        return jsonify({'error': 'No working directory'}), 400
    
    output_path = get_work_path(OUTPUT_FOLDER)
    merged_path = get_work_path(MERGED_FOLDER)
    
    deleted = False
    for base_path in [output_path, merged_path]:
        model_path = os.path.join(base_path, model_name)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            deleted = True
    
    if deleted:
        return jsonify({'success': True})
    
    return jsonify({'error': 'Model not found'}), 404

@app.route('/api/set_working_dir', methods=['POST'])
@log_request
def api_set_working_dir():
    global WORK_DIR
    data = request.get_json(silent=True) or {}
    wd = data.get('working_dir')
    
    if not wd:
        return jsonify({'success': False, 'error': 'No directory provided'}), 400
    
    wd = os.path.expanduser(wd)
    wd = os.path.abspath(wd)
    
    if os.path.exists(wd) and os.path.isdir(wd):
        WORK_DIR = wd
        save_config(WORK_DIR)
        for folder in [UPLOAD_FOLDER, MODELS_FOLDER, OUTPUT_FOLDER, MERGED_FOLDER]:
            os.makedirs(os.path.join(WORK_DIR, folder), exist_ok=True)
        return jsonify({'success': True, 'working_dir': WORK_DIR})
    else:
        return jsonify({'success': False, 'error': f'Directory does not exist: {wd}'}), 400

@app.route('/api/gpu_info', methods=['GET'])
@log_request
def gpu_info():
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines and lines[0].strip():
                gpu, vram = lines[0].split(',')
                return jsonify({'gpu': gpu.strip(), 'vram': vram.strip()})
    except Exception as e:
        print(f"⚠️ Error getting GPU info: {e}")
    
    return jsonify({'gpu': 'N/A', 'vram': 'N/A'})

@app.route('/api/check_deps', methods=['GET'])
@log_request
def check_deps():
    """Verifică dependențele sistem pentru GGUF export"""
    import subprocess

    def check_pkg(name):
        try:
            # Check if package is installed on Debian/Ubuntu systems
            result = subprocess.run(['dpkg', '-l', name], capture_output=True, text=True, timeout=2)
            return 'ii' in result.stdout
        except:
            return False

    deps = {
        'cmake': shutil.which('cmake') is not None,
        'gcc': shutil.which('gcc') is not None,
        'g++': shutil.which('g++') is not None,
        'make': shutil.which('make') is not None,
        'git': shutil.which('git') is not None,
        'build-essential': check_pkg('build-essential'),
        'libgomp1': check_pkg('libgomp1'),
        'libcurl4-openssl-dev': check_pkg('libcurl4-openssl-dev')
    }
    return jsonify(deps)

@app.route('/api/upload/progress', methods=['GET'])
@log_request
def upload_progress():
    """Endpoint pentru status upload (simulat)"""
    return jsonify({'progress': 0, 'status': 'idle'})

@app.route('/api/training/status/<session_id>', methods=['GET'])
@log_request
def training_status(session_id):
    """Verifică statusul antrenării"""
    if session_id in active_threads:
        thread = active_threads[session_id]
        if thread.is_alive():
            return jsonify({'status': 'running', 'session_id': session_id})
        else:
            return jsonify({'status': 'completed', 'session_id': session_id})
    return jsonify({'status': 'unknown', 'session_id': session_id})

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Returnează ultimele log-uri din buffer"""
    return jsonify({
        'logs': training_logs_buffer,
        'active_sessions': list(active_threads.keys())
    })

@app.route('/api/merge/status/<session_id>', methods=['GET'])
@log_request
def merge_status(session_id):
    """Verifică statusul merge-ului"""
    if session_id in active_threads:
        thread = active_threads[session_id]
        if thread.is_alive():
            return jsonify({'status': 'running', 'session_id': session_id})
        else:
            return jsonify({'status': 'completed', 'session_id': session_id})
    return jsonify({'status': 'unknown', 'session_id': session_id})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found', 'path': request.path}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed', 'method': request.method, 'path': request.path}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'details': str(error)}), 500

# HTML pages
@app.route('/')
def index():
    return render_template('unsloth_index.html', working_dir=WORK_DIR)

@app.route('/config')
def config_page():
    return render_template('unsloth_config.html', working_dir=WORK_DIR)

@app.route('/health')
def health():
    status = {
        'status': 'ok',
        'working_dir': WORK_DIR,
        'socketio': 'running',
        'active_threads': len(active_threads)
    }
    if WORK_DIR:
        status['folders'] = {
            'uploads': os.path.exists(get_work_path(UPLOAD_FOLDER)),
            'outputs': os.path.exists(get_work_path(OUTPUT_FOLDER)),
            'merged': os.path.exists(get_work_path(MERGED_FOLDER))
        }
    return jsonify(status)

if __name__ == '__main__':
    WORK_DIR = load_config()
    
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("📁 Created templates directory")
    
    print("\n" + "="*60)
    print("🚀 Unsloth WebUI starting...")
    print(f"📁 Working directory: {WORK_DIR or 'Not configured'}")
    print(f"🌐 Access at: http://localhost:7862")
    print(f"⚙️ Configure at: http://localhost:7862/config")
    print(f"🔍 Health check: http://localhost:7862/health")
    print("🔧 Debug mode: ON (Reloader disabled to prevent restarts during training)")
    print("="*60 + "\n")
    
    # debug=False and use_reloader=False are important because training generates files
    # in the working directory which can trigger the Flask reloader and kill the training process.
    socketio.run(app, host='0.0.0.0', port=7862, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
