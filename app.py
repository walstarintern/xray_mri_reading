import streamlit as st
import os
import tempfile
from PIL import Image

# Import our custom engines
from vision.router import VisionRouter
from language.ollama_client import LocalLLM
from language.prompt_builder import build_medical_prompt

# --- Page Configuration ---
st.set_page_config(page_title="Offline AI Radiology Assistant", page_icon="🩺", layout="wide")

# Initialize session state variables to remember chat and data
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
    body_part = st.selectbox("Select Body Part", ["Chest", "Bone"])
    target_language = st.selectbox("Translation Language", ["English", "Hindi", "Marathi", "Spanish", "French"])
    user_level = st.selectbox("Explanation Level", ["Patient (Simple terms)", "Medical Student", "Expert Radiologist"])

# --- Main Interface ---
st.title("🩺 Offline AI Radiology Assistant")
st.write("Upload an X-ray to generate a localized, highly accurate medical summary.")

uploaded_file = st.file_uploader("Upload X-Ray Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns([1, 2])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded {body_part} X-Ray", use_column_width=True)
        
        # Analysis Button
        if st.button("🔍 Analyze X-Ray", type="primary"):
            with st.spinner("Extracting clinical findings with Vision CNN..."):
                # Save the uploaded file temporarily so the CNN can read the path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    image.save(tmp_file.name)
                    temp_path = tmp_file.name
                
                # 1. Run Vision Routing & Analysis
                try:
                    findings = st.session_state.vision_router.route_and_analyze(temp_path, body_part)
                    st.session_state.cnn_findings = findings
                    st.success("Vision Analysis Complete!")
                except Exception as e:
                    st.error(f"Vision Engine Error: {e}")
                
                # Clean up temporary file
                os.remove(temp_path)

            if st.session_state.cnn_findings:
                with st.spinner("Generating multilingual summary with Local LLM..."):
                    # 2. Build the dynamic prompt
                    st.session_state.system_prompt = build_medical_prompt(
                        cnn_findings=st.session_state.cnn_findings,
                        target_language=target_language,
                        user_level=user_level
                    )
                    
                    # 3. Generate initial summary
                    initial_prompt = f"Please provide the initial summary for this {body_part} X-ray."
                    response = st.session_state.llm.chat(
                        system_prompt=st.session_state.system_prompt,
                        user_question=initial_prompt,
                        chat_history=[]
                    )
                    
                    # 4. Save to chat history
                    st.session_state.chat_history = [
                        {"role": "assistant", "content": response}
                    ]

    # --- Chat Interface ---
    # --- Chat Interface ---
    with col2:
        st.subheader("💬 Clinical Discussion")
        
        # 1. Create a fixed-height container that forces a scrollbar
        chat_container = st.container(height=550, border=False)
        
        # 2. Put the messages INSIDE the scrollable container
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # 3. Put the input box OUTSIDE the container so it locks to the bottom
        if prompt := st.chat_input("Ask a follow-up question about this X-ray..."):
            if not st.session_state.system_prompt:
                st.warning("Please click 'Analyze X-Ray' first!")
            else:
                # Append user question to UI
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Get LLM response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Format history for Ollama (excluding system prompt from history list)
                        history_for_llm = [
                            {"role": msg["role"], "content": msg["content"]} 
                            for msg in st.session_state.chat_history[:-1]
                        ]
                        
                        answer = st.session_state.llm.chat(
                            system_prompt=st.session_state.system_prompt,
                            user_question=prompt,
                            chat_history=history_for_llm
                        )
                        st.markdown(answer)
                        
                # Append AI answer to history
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

                # --- THE MAGIC FIX ---
                # Force a screen refresh to lock the input bar at the bottom!
                st.rerun()