import streamlit as st
import requests
import re
import pandas as pd
import os
import logging
import base64 # Needed for base64 encoding
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional # Added for type hinting

# --- Configuration ---
# Read the backend URL from an environment variable, default to localhost if not set
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip('/')
BACKEND_API_BASE_URL = f"{BACKEND_URL}/api/v1" # Using the API prefix

# --- Image Paths (Assuming an 'assets' folder in the 'frontend' directory) ---
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "images")
IMG_STREAMLIT = os.path.join(ASSETS_DIR, "streamlit.png")
IMG_FASTAPI = os.path.join(ASSETS_DIR, "fastapi.png")
IMG_GEMINI = os.path.join(ASSETS_DIR, "gemini15flash.png")
IMG_LANGCHAIN = os.path.join(ASSETS_DIR, "langchain.jpeg")
IMG_CHROMADB = os.path.join(ASSETS_DIR, "chromadb.jpeg")

# --- Initialize Logging for Frontend (Optional) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - Streamlit - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Streamlit UI Setup ---
st.set_page_config(page_title="RAG Application Assistant", layout="wide")
st.title("📑 Application Validation Assistant")
st.markdown("""
Welcome! This tool helps you work with application guidelines.
1.  **Upload Guideline PDFs**: Index the content of your guideline documents.
2.  **Upload Application PDF**: Submit an application form for automated validation against the guidelines.
3.  **Chat with Guidelines**: Ask questions about the processed guideline documents.
""")

# --- Initialize Session State Variables ---
if 'messages' not in st.session_state:
    st.session_state.messages = []  # Intended type: List[Dict[str, str]]
if 'validation_summary' not in st.session_state:
    st.session_state.validation_summary = None  # Intended type: Optional[str]
if 'guidelines_processed' not in st.session_state:
    st.session_state.guidelines_processed = False  # Intended type: bool
if 'backend_accessible' not in st.session_state:
    st.session_state.backend_accessible = None # Intended type: Optional[bool]

# --- Helper Functions ---
def styled_text(text: str, size_px: int = 14, color: Optional[str] = None) -> str:
    """
    Formats text with a specific font size and optional color for Streamlit Markdown.
    Basic HTML escaping is performed.
    """
    style = f"font-size:{size_px}px;"
    if color:
        style += f" color:{color};"
    escaped_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<span style="{style}">{escaped_text}</span>'

def display_validation_results(raw_summary: Optional[str]):
    """
    Parses the raw validation summary string from the backend and displays it
    as a structured table and summary counts in Streamlit.
    """
    if not raw_summary:
        st.info("No validation summary data to display.")
        return

    st.subheader("📝 Validation Results")

    summary_section_match = re.search(r"Summary\s*✅ Passed\s*:\s*(\d+)\s*❌ Failed\s*:\s*(\d+)\s*❓ Uncertain\s*:\s*(\d+)", raw_summary, re.DOTALL | re.IGNORECASE)

    individual_results_text = raw_summary
    if summary_section_match:
        individual_results_text = raw_summary[:summary_section_match.start()]

    blocks = re.split(r"\s*---\s*", individual_results_text.strip())

    results_data: List[Dict[str, str]] = []
    entry_pattern = re.compile(
        r"([✅❌❓])?\s*Field\s*:\s*(.*?)\s*"
        r"Value\s*:\s*(.*?)\s*"
        r"Status\s*:\s*(.*?)\s*"
        r"Reason\s*:\s*([\s\S]*?)"
        r"(?=\s*(?:[✅❌❓]\s*Field\s*:|\Z))",
        re.IGNORECASE | re.DOTALL
    )

    for block_content in blocks:
        block_content = block_content.strip()
        if not block_content: continue

        match = entry_pattern.search(block_content)
        if match:
            icon_char, field, value, status, reason = [m.strip() if m else "" for m in match.groups()]

            current_icon = icon_char if icon_char else "❓"
            if "passed" in status.lower() and current_icon != "✅": current_icon = "✅"
            elif "failed" in status.lower() and current_icon != "❌": current_icon = "❌"
            elif "uncertain" in status.lower() and current_icon not in ["✅", "❌"]: current_icon = "❓"


            results_data.append({
                "Field": field,
                "Value": value,
                "Status": f"{current_icon} {status}",
                "Reason": reason.replace('\n', ' ')
            })

    if results_data:
        df = pd.DataFrame(results_data)
        if not df.empty:
            df.insert(0, 'No.', range(1, 1 + len(df)))

        st.markdown(
            """
            <style>
                .dataframe table { border-collapse: separate; border-spacing: 0; border: 1px solid #E0E0E0; border-radius: 8px; overflow: hidden; font-size: 13px; width: 100%;}
                .dataframe th { background-color: #F5F5F5; color: #333; font-size: 14px; padding: 8px 10px; border-bottom: 1px solid #E0E0E0; text-align: left;}
                .dataframe td { padding: 8px 10px; border-bottom: 1px solid #F0F0F0;  word-break: break-word; }
                .dataframe tr:last-child td { border-bottom: none; }
                .dataframe th:nth-child(1), .dataframe td:nth-child(1) { width: 5%; }
                .dataframe th:nth-child(2), .dataframe td:nth-child(2) { width: 20%; }
                .dataframe th:nth-child(3), .dataframe td:nth-child(3) { width: 20%; }
                .dataframe th:nth-child(4), .dataframe td:nth-child(4) { width: 15%; }
                .dataframe th:nth-child(5), .dataframe td:nth-child(5) { width: 40%; }
            </style>
            """, unsafe_allow_html=True)
        st.markdown(df[["No.", "Field", "Value", "Status", "Reason"]].to_html(escape=False, index=False, classes="dataframe"), unsafe_allow_html=True)

    if summary_section_match:
        passed_count = int(summary_section_match.group(1))
        failed_count = int(summary_section_match.group(2))
        uncertain_count = int(summary_section_match.group(3))
        st.markdown("---")
        st.markdown("### Overall Summary")
        summary_cols = st.columns(3)
        summary_cols[0].metric("✅ Passed", passed_count)
        summary_cols[1].metric("❌ Failed", failed_count)
        summary_cols[2].metric("❓ Uncertain", uncertain_count)
    elif results_data:
        passed_count = sum(1 for r in results_data if "✅" in r["Status"])
        failed_count = sum(1 for r in results_data if "❌" in r["Status"])
        uncertain_count = sum(1 for r in results_data if "❓" in r["Status"])
        st.markdown("---")
        st.markdown("### Overall Summary (Calculated)")
        summary_cols = st.columns(3)
        summary_cols[0].metric("✅ Passed", passed_count)
        summary_cols[1].metric("❌ Failed", failed_count)
        summary_cols[2].metric("❓ Uncertain", uncertain_count)

# --- Backend Health Check ---
def check_backend_health():
    if st.session_state.backend_accessible is None:
        try:
            response = requests.get(f"{BACKEND_URL}/health", timeout=5)
            if response.status_code == 200 and response.json().get("status") == "healthy":
                st.session_state.backend_accessible = True
            else:
                st.session_state.backend_accessible = False
        except requests.exceptions.ConnectionError:
            st.session_state.backend_accessible = False
        except requests.exceptions.Timeout:
            st.session_state.backend_accessible = False

    if st.session_state.backend_accessible is False:
        st.error(f"🚨 Cannot connect to the backend API at {BACKEND_URL}. Please ensure the backend server is running and accessible.")
    # elif st.session_state.backend_accessible is True and not st.sidebar.button("Re-check Backend"):
    #     st.sidebar.success(f"Backend connected at {BACKEND_URL}")
    #     pass

check_backend_health()


# --- Main UI Sections ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Process Guidelines")
    uploaded_guideline_files = st.file_uploader(
        "Upload Guideline PDF(s)", type="pdf", accept_multiple_files=True,
        key="guideline_uploader", disabled=st.session_state.backend_accessible is False
    )

    if uploaded_guideline_files:
        st.write(f"{len(uploaded_guideline_files)} guideline file(s) ready for processing:")
        for file in uploaded_guideline_files: st.caption(f"- {file.name}")

        if st.button("Process Selected Guidelines", key="process_guidelines_button", type="primary", disabled=st.session_state.backend_accessible is False):
            files_to_send = [('files', (file.name, file.getvalue(), 'application/pdf')) for file in uploaded_guideline_files]
            with st.spinner("Processing guidelines... This may take a moment."):
                try:
                    response = requests.post(f"{BACKEND_API_BASE_URL}/upload_guidelines/", files=files_to_send, timeout=120)
                    response.raise_for_status()
                    backend_message = response.json().get("message", "Guidelines processed.")
                    st.success(f"✅ {backend_message}")
                    st.session_state.validation_summary = None
                    st.session_state.guidelines_processed = True
                except requests.exceptions.RequestException as e:
                    err_detail = str(e)
                    if hasattr(e, 'response') and e.response is not None:
                        try: err_detail = e.response.json().get("detail", str(e))
                        except: pass
                    st.error(f"Error processing guidelines: {err_detail}")


with col2:
    st.subheader("2. Validate Application")
    uploaded_application_file = st.file_uploader(
        "Upload Application PDF", type=["pdf"], key="application_uploader",
        disabled=not st.session_state.guidelines_processed or st.session_state.backend_accessible is False
    )

    if uploaded_application_file:
        st.caption(f"Selected for validation: {uploaded_application_file.name}")

    if st.button("Validate Application Form", key="validate_application_button", type="primary",
                 disabled=not uploaded_application_file or not st.session_state.guidelines_processed or st.session_state.backend_accessible is False):
        files_payload = {'file': (uploaded_application_file.name, uploaded_application_file.getvalue(), 'application/pdf')}
        with st.spinner(f"Validating '{uploaded_application_file.name}'... This can take some time."):
            try:
                response = requests.post(f"{BACKEND_API_BASE_URL}/validate_application/", files=files_payload, timeout=180)
                response.raise_for_status()
                data = response.json()
                st.session_state.validation_summary = data.get("validation_summary")
                if st.session_state.validation_summary: st.success("✅ Validation complete!")
                else: st.warning("⚠️ Validation finished, but no summary was returned.")
            except requests.exceptions.RequestException as e:
                err_detail = str(e)
                if hasattr(e, 'response') and e.response is not None:
                    try: err_detail = e.response.json().get("detail", str(e))
                    except: pass
                st.error(f"Error during validation: {err_detail}")
                st.session_state.validation_summary = None


st.markdown("---")
if st.session_state.validation_summary:
    display_validation_results(st.session_state.validation_summary)
    if st.button("Clear Validation Results", key="clear_validation"):
        st.session_state.validation_summary = None
        st.rerun()
elif st.session_state.guidelines_processed and st.session_state.backend_accessible is True:
    st.info("Guideline documents are processed. Upload an application PDF and click 'Validate Application Form' to see results here.")


st.markdown("---")
st.header("💬 Chat with Guidelines")
if st.session_state.backend_accessible is False:
    st.warning("Chat disabled: Backend is not accessible.")
elif not st.session_state.guidelines_processed:
    st.info("Process guideline documents first to enable the chat functionality.")
else:
    st.write("Ask questions about the guidelines that have been processed.")
    for msg in st.session_state.messages:
        avatar = "🤖" if msg["role"] == "assistant" else "🧑‍💻"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(styled_text(msg["content"], size_px=15), unsafe_allow_html=True)

    if prompt := st.chat_input("Your question...", disabled=not st.session_state.guidelines_processed or st.session_state.backend_accessible is False):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"): st.markdown(styled_text(prompt, size_px=15), unsafe_allow_html=True)

        with st.spinner("Thinking..."):
            try:
                response = requests.post(f"{BACKEND_API_BASE_URL}/chat_messages/", json={"message": prompt}, timeout=60)
                response.raise_for_status()
                assistant_response = response.json().get("response", "Sorry, I couldn't get a response.")
            except requests.exceptions.RequestException as e:
                err_detail = str(e)
                if hasattr(e, 'response') and e.response is not None:
                    try: err_detail = e.response.json().get("detail", str(e))
                    except: pass
                assistant_response = f"Chat Error: {err_detail}"

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant", avatar="🤖"): st.markdown(styled_text(assistant_response, size_px=15), unsafe_allow_html=True)

        st.rerun() # Usually not needed as Streamlit reruns on input change


# --- Technologies Used Section ---
st.markdown("---")
st.subheader("Powered By")

st.markdown("""
<style>
    .tech-logo-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .tech-logo-container img {
        max-width: 100px;
        max-height: 60px;
        object-fit: contain;
        margin-bottom: 5px;
    }
    .tech-logo-container p {
        font-size: 0.8em;
        text-align: center;
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)

def get_image_base64(image_path: str) -> Optional[str]:
    """Reads an image file and returns its base64 encoded string."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error reading image {image_path}: {e}")
        return None

tech_cols = st.columns(5)
tech_logos_data = [
    {"img_path": IMG_STREAMLIT, "caption": "Streamlit", "ext": "png"},
    {"img_path": IMG_FASTAPI, "caption": "FastAPI", "ext": "png"},
    {"img_path": IMG_GEMINI, "caption": "Gemini", "ext": "png"},
    {"img_path": IMG_LANGCHAIN, "caption": "LangChain", "ext": "jpeg"},
    {"img_path": IMG_CHROMADB, "caption": "ChromaDB", "ext": "jpeg"}
]

for i, tech in enumerate(tech_logos_data):
    with tech_cols[i]:
        if os.path.exists(tech["img_path"]):
            b64_image = get_image_base64(tech["img_path"])
            if b64_image:
                st.markdown(f"""
                <div class="tech-logo-container">
                    <img src="data:image/{tech['ext']};base64,{b64_image}" alt="{tech['caption']}">
                    <p>{tech['caption']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="tech-logo-container">
                      <p style="color:orange;">{tech['caption']} (Error loading logo)</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="tech-logo-container">
                <p style="color:red;">{tech['caption']}</p>
                <p style="font-size:0.7em; color:red;">(logo not found)</p>
            </div>
            """, unsafe_allow_html=True)
            # logger.warning(f"Logo for {tech['caption']} not found at: {tech['img_path']}")


# --- Footer ---
st.markdown("---")
if st.session_state.backend_accessible is True:
    st.caption(f"Backend API connected: {BACKEND_URL} (using prefix /api/v1)")
elif st.session_state.backend_accessible is False:
     st.caption(f"Attempted to connect to backend: {BACKEND_URL} - FAILED")
else:
    st.caption(f"Backend API configured at: {BACKEND_URL} (status unknown)")
