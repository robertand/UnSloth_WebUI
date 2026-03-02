#!/usr/bin/env python3
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

# Decorator pentru logging API
def log_request(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        print(f"📡 API Request: {request.method} {request.path}")
        print(f"📦 Headers: {dict(request.headers)}")
        if request.method == 'POST':
            print(f"📦 Data: {request.get_json()}")
        try:
            response = f(*args, **kwargs)
            print(f"✅ API Response: {request.path} - Status OK")
            return response
        except Exception as e:
            print(f"❌ API Error: {request.path} - {str(e)}")
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
        print(f"🧵 Training thread {self.session_id} started for SID {self.sid}")
        try:
            # Verifică dacă e model custom
            model_name = self.config['model_name']
            if model_name.startswith('custom:'):
                model_path = model_name.replace('custom:', '')
                if not os.path.exists(model_path):
                    socketio.emit('training_output', {
                        'data': f"❌ Model path not found: {model_path}\n"
                    }, room=self.sid)
                    socketio.emit('training_complete', {
                        'success': False,
                        'error': 'Model path not found'
                    }, room=self.sid)
                    return
                model_name = model_path
            
            # Crează scriptul de antrenare
            script_path = os.path.join(WORK_DIR, f'train_script_{self.session_id}.py')
            with open(script_path, 'w') as f:
                f.write(self.generate_training_script(model_name))
            
            socketio.emit('training_output', {
                'data': "🚀 Starting training with configuration:\n"
            }, room=self.sid)
            socketio.emit('training_output', {
                'data': json.dumps(self.config, indent=2) + "\n"
            }, room=self.sid)
            
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
            
            for line in self.process.stdout:
                if line.startswith('PROGRESS_JSON: '):
                    try:
                        progress_data = json.loads(line.replace('PROGRESS_JSON: ', ''))
                        socketio.emit('training_progress', progress_data, room=self.sid)
                    except Exception as e:
                        print(f"Error parsing progress JSON: {e}")
                        socketio.emit('training_output', {'data': line}, room=self.sid)
                else:
                    socketio.emit('training_output', {'data': line}, room=self.sid)
            
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
                }, room=self.sid)
            else:
                socketio.emit('training_complete', {
                    'success': False,
                    'error': f'Training failed with return code {self.process.returncode}'
                }, room=self.sid)
                
        except Exception as e:
            socketio.emit('training_complete', {
                'success': False,
                'error': str(e)
            }, room=self.sid)
    
    def generate_training_script(self, model_name):
        return f'''import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer
import json
import traceback

class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs:
            progress_data = {{
                "step": state.global_step,
                "max_steps": state.max_steps,
                "epoch": round(state.epoch, 2) if state.epoch is not None else None,
                "loss": logs.get("loss"),
                "learning_rate": logs.get("learning_rate"),
            }}
            if state.max_steps > 0:
                progress_data["progress"] = round((state.global_step / state.max_steps) * 100, 2)
            print(f"PROGRESS_JSON: {{json.dumps(progress_data)}}", flush=True)

try:
    # Configurare
    model_name = "{model_name}"
    dataset_path = "{self.config['dataset_path']}"
    output_dir = "{self.config['output_dir']}"
    max_seq_length = {self.config.get('max_seq_length', 2048)}
    load_in_4bit = {self.config.get('load_in_4bit', True)}
    batch_size = {self.config.get('batch_size', 2)}
    gradient_accumulation = {self.config.get('gradient_accumulation', 4)}
    learning_rate = {self.config.get('learning_rate', 2e-4)}
    num_epochs = {self.config.get('num_epochs', 3)}
    warmup_steps = {self.config.get('warmup_steps', 5)}
    save_steps = {self.config.get('save_steps', 50)}
    logging_steps = {self.config.get('logging_steps', 10)}
    lora_r = {self.config.get('lora_r', 16)}
    lora_alpha = {self.config.get('lora_alpha', 16)}
    lora_dropout = {self.config.get('lora_dropout', 0)}
    use_gradient_checkpointing = {self.config.get('use_gradient_checkpointing', True)}
    optim = "{self.config.get('optim', 'adamw_8bit')}"

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

    def format_prompt(example):
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output = example.get('output', '')
        
        if input_text:
            text = f"### Instruction:\\n{{instruction}}\\n\\n### Input:\\n{{input_text}}\\n\\n### Response:\\n{{output}}"
        else:
            text = f"### Instruction:\\n{{instruction}}\\n\\n### Response:\\n{{output}}"
        
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
        packing=True,
        callbacks=[ProgressCallback()],
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
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

class MergeThread(threading.Thread):
    def __init__(self, sid, base_model_path, lora_path, output_path, merge_method):
        threading.Thread.__init__(self)
        self.sid = sid
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.output_path = output_path
        self.merge_method = merge_method
        self.daemon = True
        self.process = None
        self.session_id = str(int(time.time()))
        
    def run(self):
        print(f"🧵 Merge thread {self.session_id} started for SID {self.sid}")
        try:
            script_path = os.path.join(WORK_DIR, f'merge_script_{self.session_id}.py')
            with open(script_path, 'w') as f:
                f.write(self.generate_merge_script())
            
            socketio.emit('merge_output', {
                'data': "🔄 Starting LoRA merge process...\n"
            }, room=self.sid)
            
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
                socketio.emit('merge_output', {'data': line}, room=self.sid)
            
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
                }, room=self.sid)
            else:
                socketio.emit('merge_complete', {
                    'success': False,
                    'error': f'Merge failed with return code {self.process.returncode}'
                }, room=self.sid)
                
        except Exception as e:
            socketio.emit('merge_complete', {
                'success': False,
                'error': str(e)
            }, room=self.sid)
    
    def generate_merge_script(self):
        return f'''import os
import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import shutil
import json
import traceback
import time

try:
    base_model = "{self.base_model_path}"
    lora_path = "{self.lora_path}"
    output_path = "{self.output_path}"
    merge_method = "{self.merge_method}"

    print(f"📂 Base model: {{base_model}}")
    print(f"📂 LoRA adapters: {{lora_path}}")
    print(f"📂 Output: {{output_path}}")

    print("🚀 Loading base model and LoRA adapters...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=4096,
            load_in_4bit=(merge_method == "merged_4bit"),
            device_map="auto",
        )
        print("✅ Base model loaded")
        
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        print("✅ LoRA adapters loaded")
        
    except Exception as e:
        print(f"❌ Error loading model: {{e}}")
        traceback.print_exc()
        raise

    print(f"🔄 Merging LoRA with base model using method: {{merge_method}}...")
    try:
        merged_model = model.merge_and_unload()
        print("✅ Model merged successfully")
        
        print("💾 Saving merged model...")
        os.makedirs(output_path, exist_ok=True)
        
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
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
        emit('connected', {'working_dir': WORK_DIR})

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
    data = request.json
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
        return '', 200
        
    if not WORK_DIR:
        return jsonify({'error': 'No working directory'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    upload_path = get_work_path(UPLOAD_FOLDER)
    filepath = os.path.join(upload_path, filename)
    
    # Salvează fișierul
    file.save(filepath)
    print(f"📁 File uploaded: {filename}")
    
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

@app.route('/api/start_training', methods=['POST'])
@log_request
def start_training():
    if not WORK_DIR:
        return jsonify({'error': 'No working directory'}), 400
    
    data = request.get_json()
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
        'lora_r': data.get('lora_r', 16),
        'load_in_4bit': data.get('load_in_4bit', True),
    }
    
    # Pornește thread-ul de antrenare
    thread = TrainingThread(socket_id, config)
    thread.start()
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
    
    data = request.get_json()
    print(f"📦 Received merge data: {json.dumps(data, indent=2)}")
    
    socket_id = data.get('socket_id')
    trained_model = data.get('trained_model')
    merge_method = data.get('merge_method', 'merged_16bit')
    
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
    
    thread = MergeThread(socket_id, base_model, lora_path, output_path, merge_method)
    thread.start()
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
    data = request.json
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
    print(f"🔧 Debug mode: ON")
    print("="*60 + "\n")
    
    socketio.run(app, host='0.0.0.0', port=7862, debug=True, allow_unsafe_werkzeug=True)
