# 🩺 Offline AI Radiology Assistant

An **entirely offline**, locally-run AI assistant for reading and interpreting X-Ray and MRI scans. It combines a CNN-based vision engine, a local LLM (via Ollama), and a translation microservice to deliver structured medical summaries in English, Hindi, or Marathi — all on your own machine with no cloud API required.

---

## ✨ Features

- **Multi-scan support** — handles both X-Ray and MRI images
- **Body-part routing** — intelligently routes scans to the right vision model based on body part (Chest, Bone, Brain, Knee, Spine)
- **Local LLM summaries** — uses `qwen2.5:3b` via Ollama to generate clinical summaries from vision findings
- **Multilingual output** — translates results into English, Hindi, or Marathi via a local Flask translation server
- **Explanation levels** — tailor output for Patients (simple terms), Medical Students, or Expert Radiologists
- **Chat interface** — ask follow-up questions about the scan in an interactive chat window
- **100% offline** — no data ever leaves your machine

---

## 🗂️ Project Structure

```
xray_mri_reading/
│
├── app.py                      # Main Streamlit application
├── translation_server.py       # Flask microservice for language translation
│
├── requirements.txt            # Dependencies for app.py environment
├── translator_requirements.txt # Dependencies for translation_server environment
│
├── vision/
│   └── router.py               # Routes scans to the correct CNN model by type & body part
│
├── language/
│   ├── ollama_client.py        # Wrapper for local Ollama LLM (qwen2.5:3b)
│   └── prompt_builder.py       # Builds structured medical prompts from vision findings
│
├── utils/                      # Utility/helper modules
├── test/                       # Test scripts
└── local_chats.json            # Local chat history storage
```

---

## 🧠 How It Works

1. **Upload** an X-Ray or MRI image (JPG/PNG) through the Streamlit UI.
2. **Vision Engine** (`vision/router.py`) routes the image to the appropriate CNN model based on scan type and body part, extracting raw clinical findings.
3. **Prompt Builder** (`language/prompt_builder.py`) wraps those findings into a structured medical prompt.
4. **Local LLM** (`language/ollama_client.py`) — powered by `qwen2.5:3b` running through Ollama — generates a detailed summary.
5. **Translation Server** (`translation_server.py`) translates the output into the selected language (Hindi or Marathi) via a local Flask API on `http://127.0.0.1:5000`.
6. The result is displayed in a chat interface where you can ask follow-up questions.

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| Vision Analysis | PyTorch, TorchXRayVision, timm |
| LLM | Ollama + qwen2.5:3b |
| Translation | Flask + googletrans |
| Image Processing | Pillow, OpenCV, albumentations |

---

## 🚀 Quick Setup

See **[SETUP.md](./SETUP.md)** for the full step-by-step setup guide.

The app requires **three separate terminals** running simultaneously:
- **Terminal 1** — Ollama server
- **Terminal 2** — Translation microservice
- **Terminal 3** — Streamlit app

---

## ⚠️ Disclaimer

This tool is intended for **educational and research purposes only**. It is not a substitute for professional medical diagnosis. Always consult a qualified radiologist or physician for clinical decisions.
