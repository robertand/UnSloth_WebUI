#!/bin/bash
echo "🚀 Starting Unsloth WebUI..."
echo "📁 Creating necessary directories..."

# Crează directoarele necesare
mkdir -p templates
mkdir -p WORK/uploads WORK/models WORK/outputs WORK/merged_models

# Verifică dacă fișierele template există
if [ ! -f "templates/unsloth_index.html" ]; then
    echo "❌ Error: templates/unsloth_index.html not found!"
    exit 1
fi

if [ ! -f "templates/unsloth_config.html" ]; then
    echo "❌ Error: templates/unsloth_config.html not found!"
    exit 1
fi

echo "✅ All files present"
echo "🌐 Starting server on http://localhost:7862"
echo "⚙️ Configure at http://localhost:7862/config"
echo "🔍 Health check at http://localhost:7862/health"
echo ""

# Setează variabile de mediu pentru Python
export PYTHONUNBUFFERED=1

# Pornește aplicația
python unsloth_webui.py
