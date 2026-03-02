#!/usr/bin/env python3
import requests
import json
import sys
import time

BASE_URL = "http://localhost:7862"

def check_endpoint(name, url, method='GET', data=None):
    print(f"\n🔍 Checking {name}... ", end="")
    try:
        if method == 'GET':
            r = requests.get(f"{BASE_URL}{url}", timeout=5)
        else:
            r = requests.post(f"{BASE_URL}{url}", json=data, timeout=5)
            
        if r.status_code == 200:
            print(f"✅ {r.status_code}")
            try:
                response_data = r.json()
                print(f"   Response type: {type(response_data)}")
                if isinstance(response_data, list):
                    print(f"   Items: {len(response_data)}")
                elif isinstance(response_data, dict):
                    print(f"   Keys: {list(response_data.keys())}")
            except:
                print(f"   Response: {r.text[:100]}...")
        else:
            print(f"❌ {r.status_code}")
            print(f"   Error: {r.text[:200]}")
    except Exception as e:
        print(f"❌ Failed: {e}")

def main():
    print("="*60)
    print("🔍 Checking Unsloth WebUI API endpoints")
    print("="*60)
    
    # Verifică dacă serverul rulează
    print("\n📡 Testing server connection...")
    try:
        start_time = time.time()
        r = requests.get(f"{BASE_URL}/health", timeout=2)
        response_time = (time.time() - start_time) * 1000
        if r.status_code == 200:
            print(f"✅ Server is running (response time: {response_time:.0f}ms)")
            health_data = r.json()
            print(f"   Working dir: {health_data.get('working_dir')}")
            print(f"   Active threads: {health_data.get('active_threads')}")
        else:
            print(f"❌ Server returned {r.status_code}")
            print(f"   Make sure the server is running on {BASE_URL}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server")
        print(f"   Make sure the server is running on {BASE_URL}")
        print("   Run './start_unsloth.sh' first")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("📋 Testing individual endpoints:")
    print("="*60)
    
    endpoints = [
        ("Health", "/health"),
        ("Models List", "/api/models"),
        ("Files List", "/api/files"),
        ("Trained Models", "/api/trained_models"),
        ("Merged Models", "/api/merged_models"),
        ("GPU Info", "/api/gpu_info"),
        ("Custom Models", "/api/custom_models"),
        ("Upload Progress", "/api/upload/progress"),
    ]
    
    for name, url in endpoints:
        check_endpoint(name, url)
    
    print("\n" + "="*60)
    print("🧪 Testing POST endpoints:")
    print("="*60)
    
    # Test start_training cu socket_id fals
    test_config = {
        "model_name": "unsloth/Qwen2.5-3B-Instruct",
        "dataset_file": "test.json",
        "socket_id": "test-socket-123"
    }
    check_endpoint("Start Training", "/api/start_training", method='POST', data=test_config)
    
    print("\n" + "="*60)
    print("✅ Check complete")
    print("="*60)

if __name__ == "__main__":
    main()
