import streamlit as st
import os
import tempfile
from PIL import Image
import requests

def get_translation(text, dest_lang):
    """Sends text to the local translation microservice."""
    if dest_lang == "en" or not text:
        return text
    try:
        response = requests.post("http://127.0.0.1:5000/translate", json={
            "text": text,
            "dest": dest_lang
        })
        if response.status_code == 200:
            return response.json().get("translated_text", text)
        else:
            return f"[Translation Error] {text}"
    except Exception:
        return f"[Translation Server Offline] {text}"

# Import our custom engines
from vision.router import VisionRouter
from language.ollama_client import LocalLLM
from language.prompt_builder import build_medical_prompt

# --- Page Configuration ---
st.set_page_config(page_title="Offline AI Radiology Assistant", page_icon="🩺", layout="wide")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "cnn_findings" not in st.session_state:
    st.session_state.cnn_findings = None
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = None
if "vision_router" not in st.session_state:
    st.session_state.vision_router = VisionRouter()
if "llm" not in st.session_state:
    st.session_state.llm = LocalLLM(model_name="qwen2.5:3b")

# --- Sidebar Controls ---
with st.sidebar:
    st.title("⚙️ System Settings")
    
    # 1. NEW: Scan Type Toggle
    scan_type = st.radio("Select Scan Type", ["X-Ray", "MRI"])
    
    # 2. NEW: Dynamic Body Part Selection
    if scan_type == "X-Ray":
        body_part = st.selectbox("Select Body Part", ["Chest", "Bone"])
    else:
        body_part = st.selectbox("Select Body Part", ["Brain", "Knee", "Spine"])
        
    target_language = st.selectbox("Translation Language", ["English", "Hindi", "Marathi"])
    
    # Map the UI selection to Google Translate codes
    lang_codes = {"English": "en", "Hindi": "hi", "Marathi": "mr"}
    dest_code = lang_codes.get(target_language, "en")

    user_level = st.selectbox("Explanation Level", ["Patient (Simple terms)", "Medical Student", "Expert Radiologist"])

# --- Main Interface ---
st.title("🩺 Offline AI Radiology Assistant")
st.write("Upload a scan to generate a localized, highly accurate medical summary.")

# Updated text to reflect both X-Rays and MRIs
uploaded_file = st.file_uploader(f"Upload {scan_type} Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns([1, 2])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded {body_part} {scan_type}", use_column_width=True)
        
        # Analysis Button
        if st.button(f"🔍 Analyze {scan_type}", type="primary"):
            with st.spinner("Extracting clinical findings with Vision AI..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    image.save(tmp_file.name)
                    temp_path = tmp_file.name
                
                # 1. Run Vision Routing & Analysis (Passing BOTH scan_type and body_part)
                try:
                    findings = st.session_state.vision_router.route_and_analyze(temp_path, scan_type, body_part)
                    st.session_state.cnn_findings = findings
                    st.success("Vision Analysis Complete!")
                    
                    st.info(f"🧠 RAW AI OUTPUT: {findings}")
                    
                except Exception as e:
                    st.error(f"Vision Engine Error: {e}")
                
                os.remove(temp_path)

            if st.session_state.cnn_findings:
                with st.spinner("Generating multilingual summary with Local LLM..."):
                    
                    # 2. Build the dynamic prompt
                    st.session_state.system_prompt = build_medical_prompt(
                        cnn_findings=st.session_state.cnn_findings,
                        scan_type=scan_type,
                        body_part=body_part,       
                        user_level=user_level
                    )
                    
                    # 3. Generate initial summary
                    initial_prompt = f"Please provide the initial summary for this {body_part} {scan_type}."
                    
                    english_response = st.session_state.llm.chat(
                        system_prompt=st.session_state.system_prompt,
                        user_question=initial_prompt,
                        chat_history=[]
                    )
                    
                    final_response = get_translation(english_response, dest_code)
                    
                    st.session_state.chat_history = [
                        {"role": "assistant", "content": final_response}
                    ]

    # --- Chat Interface ---
    with col2:
        st.subheader("💬 Clinical Discussion")
        
        chat_container = st.container(height=550, border=False)
        
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if prompt := st.chat_input(f"Ask a follow-up question about this {scan_type}..."):
            if not st.session_state.system_prompt:
                st.warning(f"Please click 'Analyze {scan_type}' first!")
            else:
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)

                with chat_container:
                    with st.chat_message("assistant"):
                        with st.spinner("Translating and Thinking..."):
                            english_prompt = get_translation(prompt, "en") if dest_code != "en" else prompt
                            
                            history_for_llm = [
                                {"role": msg["role"], "content": msg["content"]} 
                                for msg in st.session_state.chat_history[:-1]
                            ]
                            
                            english_answer = st.session_state.llm.chat(
                                system_prompt=st.session_state.system_prompt,
                                user_question=english_prompt,
                                chat_history=history_for_llm
                            )
                            
                            final_answer = get_translation(english_answer, dest_code)
                            st.markdown(final_answer)
                        
                st.session_state.chat_history.append({"role": "assistant", "content": final_answer})
                st.rerun()