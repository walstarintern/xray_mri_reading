# ⚙️ Setup Guide

This project requires **three separate terminals** running at the same time. Follow each section in order before moving to the next.

---

## 📋 Prerequisites

- Python 3.9 or higher
- `pip` and `venv` available
- [Ollama](https://ollama.com/download) installed locally (see Step 1)
- Git (to clone the repo)

---

## 📥 Clone the Repository

```bash
git clone https://github.com/walstarintern/xray_mri_reading.git
cd xray_mri_reading
```

---

## 🖥️ TERMINAL 1 — Ollama Server

This terminal runs the local LLM backend. It must stay open the entire time you use the app.

### Step 1: Install Ollama

Download and install Ollama for your OS from [https://ollama.com/download](https://ollama.com/download).

- **macOS / Linux**: run the installer script or use the package manager
- **Windows**: download the `.exe` installer

### Step 2: Pull the Qwen model

This downloads the `qwen2.5:3b` model used for generating medical summaries. Run this once — it may take a few minutes depending on your connection.

```bash
ollama pull qwen2.5:3b
```

### Step 3: Start the Ollama server

```bash
ollama serve
```

> ✅ You should see: `Listening on 127.0.0.1:11434`
>
> **Keep this terminal open.**

---

## 🌐 TERMINAL 2 — Translation Server

This terminal runs the Flask microservice that handles Hindi and Marathi translations. It must also stay open.

### Step 1: Create and activate a virtual environment

```bash
# Create the environment
python -m venv venv_translator

# Activate it
# On macOS / Linux:
source venv_translator/bin/activate

# On Windows:
venv_translator\Scripts\activate
```

### Step 2: Install translator dependencies

```bash
pip install -r translator_requirements.txt
```

This installs:
- `flask` — the web server framework
- `googletrans==4.0.0-rc1` — the translation library

### Step 3: Run the translation server

```bash
python translation_server.py
```

> ✅ You should see: `🚀 Translator Microservice is running on http://127.0.0.1:5000`
>
> **Keep this terminal open.**

---

## 🩺 TERMINAL 3 — Main Streamlit App

This terminal runs the actual radiology assistant UI. Open it only after Terminals 1 and 2 are running.

### Step 1: Create and activate a virtual environment

```bash
# Create the environment
python -m venv venv_app

# Activate it
# On macOS / Linux:
source venv_app/bin/activate

# On Windows:
venv_app\Scripts\activate
```

### Step 2: Install app dependencies

> ⚠️ This installs PyTorch and medical imaging libraries — the download may take several minutes.

```bash
pip install -r requirements.txt
```

This installs:
- `streamlit` — the UI framework
- `torch`, `torchvision` — deep learning backend
- `torchxrayvision` — pre-trained X-ray CNN models
- `ollama` — Python client for the local LLM
- `Pillow`, `opencv-python-headless` — image processing
- `timm`, `albumentations`, `numpy` — model and augmentation utilities

### Step 3: Run the app

```bash
streamlit run app.py
```

> ✅ The app will open automatically in your browser at `http://localhost:8501`

---

## ✅ All Three Running — What to Expect

Once all three terminals are active, you'll see the Streamlit UI in your browser. Here's the normal flow:

1. Select your **scan type** (X-Ray or MRI) and **body part** in the sidebar.
2. Choose your preferred **language** (English, Hindi, or Marathi) and **explanation level**.
3. **Upload** a JPG or PNG scan image.
4. Click **Analyze** — the vision engine processes the image and the LLM generates a summary.
5. Use the **chat box** to ask follow-up questions about the findings.

---

## 🛑 Stopping Everything

To stop the app cleanly, press `Ctrl + C` in each terminal, then deactivate the environments:

```bash
deactivate
```

---

## 🐛 Troubleshooting

**Translation returns `[Translation Server Offline]`**
Make sure Terminal 2 is running and the Flask server shows "Listening on 127.0.0.1:5000". Restart it if needed.

**LLM responses are empty or erroring**
Check Terminal 1 — the Ollama server must be running. Also confirm the model was pulled with `ollama list` (should show `qwen2.5:3b`).

**Vision engine error on upload**
Ensure all packages in `requirements.txt` installed without errors, especially `torchxrayvision` and `timm`. Try reinstalling with `pip install -r requirements.txt --force-reinstall`.

**Port conflict on 5000 or 11434**
Another process may be using that port. On macOS/Linux, find it with `lsof -i :5000` and kill it, then restart the translation server.
