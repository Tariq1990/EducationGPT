import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os
import sys
import subprocess
import shutil
import re
from openai import OpenAI
from dotenv import load_dotenv

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.local_rag import LocalRAGAdvisor, build_local_documents, load_documents

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="EduSmart AI | School Resource Dashboard",
    page_icon="School",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PWA & Mobile Optimization ---
st.markdown("""
<head>
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <link rel="apple-touch-icon" href="https://cdn-icons-png.flaticon.com/512/167/167707.png">
</head>
""", unsafe_allow_html=True)

# --- Premium Look & Feel (Custom CSS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Outfit:wght@500;700&display=swap');

    :root {
        --primary-gradient: linear-gradient(135deg, #0b3d91 0%, #0ea5e9 100%);
        --glass-bg: rgba(255, 255, 255, 0.10);
        --glass-border: rgba(255, 255, 255, 0.20);
        --text-primary: #0b1324;
        --text-muted: #334155;
        --card-bg: rgba(255, 255, 255, 0.85);
        --card-border: rgba(15, 23, 42, 0.08);
    }

    * {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
    }

    /* High-contrast light theme background */
    .stApp {
        background:
            radial-gradient(1200px 600px at 10% -10%, rgba(14, 165, 233, 0.18), transparent 60%),
            radial-gradient(900px 500px at 95% 10%, rgba(11, 61, 145, 0.20), transparent 55%),
            linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        color: var(--text-primary);
    }

    .main-header {
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    /* Chat bubble styling */
    .stChatMessage {
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid var(--card-border);
        background: var(--card-bg);
    }

    .stChatMessage[data-testimonial="user"] {
        background: rgba(14, 165, 233, 0.12);
    }

    .stChatMessage[data-testimonial="assistant"] {
        background: rgba(11, 61, 145, 0.08);
    }

    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background: #0b1324;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    .sidebar-title {
        color: #f8fafc;
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem;
        margin-bottom: 2rem;
        padding-top: 1rem;
    }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif;
        color: #0b3d91 !important;
    }

    [data-testid="stMetricLabel"] {
        color: var(--text-muted);
    }

    .stMarkdown, .stText, .stCaption {
        color: var(--text-primary);
    }

    /* Chart containers */
    [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration & Models ---
api_key = os.getenv("DEEPSEEK_API_KEY")
use_local_llm = os.getenv("USE_LOCAL_LLM", "false").lower() in ("1", "true", "yes")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
ollama_models = [m.strip() for m in os.getenv("OLLAMA_MODELS", "llama3.2:1b,llama3:8b,qwen2.5:7b").split(",") if m.strip()]
if api_key:
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
else:
    client = None

local_client = OpenAI(
    api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
    base_url=ollama_base_url
)
local_rag_top_k = int(os.getenv("LOCAL_RAG_TOP_K", "5"))
local_rag_min_score = float(os.getenv("LOCAL_RAG_MIN_SCORE", "0.08"))

DATA_PATH = os.getenv("PROCESSED_CSV_PATH", os.path.join("data", "processed", "processed_school_data.csv"))
API_URL = os.getenv("PREDICT_API_URL", "http://localhost:8000/predict")
RAG_INDEX_PATH = os.getenv("RAG_INDEX_PATH", os.path.join("data", "processed", "rag_chunks.json"))
if not os.path.isabs(RAG_INDEX_PATH):
    RAG_INDEX_PATH = os.path.normpath(os.path.join(PROJECT_ROOT, RAG_INDEX_PATH))

model_choice = None
# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown('<p class="sidebar-title">EduSmart AI</p>', unsafe_allow_html=True)
    page = st.radio("Navigation", ["Dashboard & Analytics", "AI Advisor (Chat)"], label_visibility="collapsed")
    
    st.divider()
    if page == "AI Advisor (Chat)":
        st.subheader("Settings")
        if not use_local_llm and client:
            model_choice = st.selectbox(
                "AI Model",
                ["deepseek-chat", "deepseek-reasoner"],
                index=0,
                help="Select Deepseek model."
            )
        else:
            if not ollama_models:
                ollama_models = [ollama_model]
            default_index = ollama_models.index(ollama_model) if ollama_model in ollama_models else 0
            model_choice = st.selectbox(
                "Local LLM Model",
                options=ollama_models,
                index=default_index,
                help="Select local Ollama model."
            )
            st.caption(f"Local LLM: {model_choice}")
        if st.button("New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.caption("v2.2.0 | Premium Edition")

# --- Functions ---
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

def get_chat_context(df, last_prediction=None):
    context = "You are EduSmart AI, a highly intelligent Educational Resource Planning Advisor for Khyber Pakhtunkhwa (KP) schools. "
    context += "Your goal is to help administrators optimize resource allocation (teachers, classrooms, desks, fans) based on data. "
    
    if df is not None:
        total = len(df)
        critical = df['is_critical'].sum()
        avg_teachers = round(df['total_teachers'].mean(), 1)
        top_districts = df['District'].value_counts().head(3).index.tolist()
        
        context += f"\n\nSystem Knowledge Base:\n- Total Schools Analyzed: {total}\n- Critical Shortage Alerts: {critical}\n- Average Teachers per School: {avg_teachers}\n- Major Districts: {', '.join(top_districts)}."
    
    if last_prediction:
        lp = last_prediction
        context += f"\n\nActive Case Analysis:\n- School Level: {lp['LEVEL']}\n- Location: {lp['District']}\n- Prediction: {'CRITICAL RESOURCE SHORTAGE' if lp['is_critical'] == 1 else 'NORMAL STATUS'}\n- Confidence: {max(lp['probability']):.2%}\n- Infrastructure: {lp['No_of_Pakka_Class_Rooms']} classrooms, {lp['total_teachers']} teachers, PKR {lp['Presentbalance']} PTC balance."
    
    return context

def query_local_llm(chat_messages, model_name):
    response = local_client.chat.completions.create(
        model=model_name,
        messages=chat_messages,
        stream=False,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

def get_advisor_system_prompt(df, last_prediction=None):
    base_context = get_chat_context(df, last_prediction)
    return (
        f"{base_context}\n\n"
        "Instructions:\n"
        "- Use only the provided school context and user input.\n"
        "- If data is missing, state what is missing instead of guessing.\n"
        "- Never invent school names, district stats, or prediction values.\n"
        "- Keep answers practical for school administrators.\n"
        "- For recommendations, provide short actionable steps."
    )

def get_deterministic_answer(prompt, df, last_prediction=None):
    p = prompt.lower()
    if df is None:
        if any(k in p for k in ["total schools", "critical", "district", "average teachers"]):
            return "I cannot compute that right now because the processed dataset is not loaded."
        return None

    # If user asks for a specific district that is not in data, fail fast.
    district_match = re.search(r"district\s+([a-zA-Z][a-zA-Z\s\-]{1,40})", prompt, flags=re.IGNORECASE)
    if district_match and "District" in df.columns:
        requested = district_match.group(1).strip().lower()
        known = {str(d).strip().lower() for d in df["District"].dropna().unique().tolist()}
        if requested and requested not in known:
            available = ", ".join(sorted({str(d).strip() for d in df["District"].dropna().unique().tolist()}))
            return f"District '{district_match.group(1).strip()}' is not available in local data. Available district(s): {available}."

    if any(k in p for k in ["total schools", "how many schools", "number of schools"]):
        return f"Total schools analyzed: {len(df)}."

    if any(k in p for k in ["critical schools", "critical count", "how many critical"]):
        critical = int(df["is_critical"].sum())
        pct = (critical / len(df)) * 100 if len(df) else 0
        return f"Critical schools: {critical} out of {len(df)} ({pct:.1f}%)."

    if "average teachers" in p or "avg teachers" in p:
        return f"Average teachers per school: {df['total_teachers'].mean():.1f}."

    if "top district" in p or "most schools" in p:
        top = df["District"].value_counts().idxmax()
        count = int(df["District"].value_counts().iloc[0])
        return f"District with most surveyed schools: {top} ({count} schools)."

    if any(k in p for k in ["last prediction", "latest prediction", "prediction result"]):
        if not last_prediction:
            return "No recent prediction is available. Generate one in 'Dashboard & Analytics' first."
        conf = max(last_prediction.get("probability", [0, 0]))
        status = "CRITICAL RESOURCE SHORTAGE" if last_prediction.get("is_critical") == 1 else "NORMAL STATUS"
        return (
            f"Last prediction for {last_prediction.get('District', 'N/A')} ({last_prediction.get('LEVEL', 'N/A')}): "
            f"{status}, confidence {conf:.2%}."
        )

    return None

def discover_ollama_models(default_models):
    if not shutil.which("ollama"):
        return default_models
    try:
        result = subprocess.run(
            ["ollama", "list"],
            text=True,
            capture_output=True,
            timeout=10
        )
        if result.returncode != 0:
            return default_models
        lines = result.stdout.splitlines()
        discovered = []
        for line in lines[1:]:
            parts = line.split()
            if parts:
                discovered.append(parts[0].strip())
        # De-duplicate while preserving order
        return list(dict.fromkeys(discovered + default_models)) if discovered else default_models
    except Exception:
        return default_models

def build_local_rag(df):
    docs = load_documents(RAG_INDEX_PATH)
    if not docs and df is not None:
        docs = build_local_documents(df)
    if not docs:
        return None
    return LocalRAGAdvisor(docs)

def format_rag_context(chunks):
    lines = []
    for c in chunks:
        lines.append(f"[{c.chunk_id}] source={c.source} score={c.score:.3f}\n{c.text}")
    return "\n\n".join(lines)

def boost_district_chunk(query, df, rag, chunks):
    if df is None or rag is None or "District" not in df.columns:
        return chunks
    q = (query or "").lower()
    districts = [str(d).strip() for d in df["District"].dropna().unique().tolist()]
    for district in districts:
        if district and district.lower() in q:
            chunk = rag.get_chunk_by_id(f"district-{district}")
            if chunk and all(c.chunk_id != chunk.chunk_id for c in chunks):
                return [chunk] + chunks
            break
    return chunks

# --- Main Page Execution ---
df = load_data()
if "local_rag" not in st.session_state:
    st.session_state.local_rag = build_local_rag(df)
if st.session_state.get("local_rag") is None and df is not None:
    st.session_state.local_rag = build_local_rag(df)

# Client is already configured globally if API key exists

if page == "Dashboard & Analytics":
    st.markdown('<h1 class="main-header">School Resource Insights</h1>', unsafe_allow_html=True)
    
    if df is not None:
        # Filters in columns for cleaner dashboard
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            district_list = sorted(df['District'].unique())
            district = st.multiselect("Filter District", options=district_list, default=district_list[:3])
        with col_f2:
            level_list = sorted(df['LEVEL'].unique())
            level = st.multiselect("Filter Level", options=level_list, default=level_list)
        with col_f3:
            gender_list = sorted(df['Gender'].dropna().unique())
            gender = st.multiselect("Filter Gender", options=gender_list, default=gender_list)

        filtered_df = df[
            (df['District'].isin(district)) &
            (df['LEVEL'].isin(level)) &
            (df['Gender'].isin(gender))
        ]
        filtered_df = filtered_df.copy()
        for col in ["Covered_Area", "Presentbalance", "total_teachers"]:
            if col in filtered_df.columns:
                filtered_df[col] = pd.to_numeric(filtered_df[col], errors="coerce").fillna(0)

        # KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Schools", len(filtered_df))
        kpi2.metric("Critical Status", filtered_df['is_critical'].sum(), delta=f"{int(filtered_df['is_critical'].mean()*100)}% risk")
        kpi3.metric("Avg Teachers", round(filtered_df['total_teachers'].mean(), 1))
        kpi4.metric("Avg PTC Balance", f"PKR {round(filtered_df['Presentbalance'].mean(), 0):,}")

        # Visuals
        st.divider()
        v1, v2 = st.columns(2)
        with v1:
            fig1 = px.histogram(filtered_df, x="District", color="is_critical", barmode="group",
                                title="Shortage Alerts by District",
                                color_discrete_map={0: "#10b981", 1: "#ef4444"},
                                template="plotly_dark")
            st.plotly_chart(fig1, use_container_width=True)
        with v2:
            fig2 = px.scatter(filtered_df, x="Presentbalance", y="total_teachers", color="LEVEL",
                             size="Covered_Area", title="Funding vs Staffing Correlation",
                             template="plotly_dark", hover_name="District")
            st.plotly_chart(fig2, use_container_width=True)

        v3, v4 = st.columns(2)
        with v3:
            fig3 = px.histogram(filtered_df, x="Gender", color="is_critical", barmode="group",
                                title="Shortage Alerts by Gender",
                                color_discrete_map={0: "#10b981", 1: "#ef4444"},
                                template="plotly_dark")
            st.plotly_chart(fig3, use_container_width=True)
        with v4:
            fig4 = px.box(filtered_df, x="Gender", y="total_teachers", color="Gender",
                          title="Teacher Distribution by Gender",
                          template="plotly_dark")
            st.plotly_chart(fig4, use_container_width=True)

        with st.expander("Filtered Data Preview"):
            preview_cols = [
                "District", "Gender", "LEVEL", "total_teachers",
                "Presentbalance", "No of Pakka Class Rooms", "is_critical"
            ]
            available_cols = [c for c in preview_cols if c in filtered_df.columns]
            st.dataframe(filtered_df[available_cols], use_container_width=True, height=320)
            
        # Prediction Form (Collapsible)
        with st.expander("Infrastructure Shortage Prediction Tool"):
            st.markdown("Enter school stats to predict if it will fall into the 'Critical' category.")
            with st.form("pred_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    if df is not None and "Gender" in df.columns:
                        gender_options = sorted(df["Gender"].dropna().unique())
                    else:
                        gender_options = ["Boys", "Girls"]
                    gender = st.selectbox("Gender", gender_options)
                    level_in = st.selectbox("Level", ["Primary", "Middle", "High", "Higher Secondary"])
                    dist_in = st.text_input("District Name", "PESHAWAR")
                with c2:
                    teachers = st.number_input("Total Teachers", min_value=0, value=3)
                    pakka_rooms = st.number_input("Pakka Class Rooms", min_value=0, value=5)
                    ptc_bal = st.number_input("Current PTC Balance", min_value=0, value=1500)
                with c3:
                    covered = st.number_input("Covered Area (sq ft)", min_value=0, value=200)
                    desks = st.number_input("Useable Desks", min_value=0, value=20)
                    fans = st.number_input("Useable Fans", min_value=0, value=10)
                
                if st.form_submit_button("Generate Prediction", use_container_width=True):
                    payload = {
                        "Gender": gender, "LEVEL": level_in, "District": dist_in.upper(),
                        "BuildingOwnerShip": "Govt", "TotalLand_Allocated": 500,
                        "Covered_Area": covered, "UnCovered_Area": 300,
                        "No_of_Pakka_Class_Rooms": pakka_rooms, "No_Of_Pakka_Other_Rooms": 2,
                        "Desks_Two_Seater_Useable": desks, "Tablet_Chairs_Useable": 50,
                        "Fans_Useable": fans, "total_teachers": teachers,
                        "avg_bps_teachers": 14, "qualified_teachers": 1,
                        "total_non_teachers": 2, "Presentbalance": ptc_bal,
                        "Fund_Recieved_from_Govt_in_This_Year": 2000
                    }
                    try:
                        response = requests.post(API_URL, json=payload, timeout=5)
                        if response.status_code == 200:
                            res = response.json()
                            st.session_state['last_prediction'] = {**payload, **res}
                            if res['is_critical'] == 1:
                                st.error(f"CRITICAL SHORTAGE DETECTED (Probability: {res['probability'][1]:.2%})")
                            else:
                                st.success(f"NORMAL STATUS (Probability: {res['probability'][0]:.2%})")
                            if "threshold" in res:
                                st.caption(f"Decision threshold in use: {res['threshold']:.2f}")
                        else: st.error(f"API Error: {response.text}")
                    except: st.error("Connection Error: Is the API server running?")

    else:
        st.warning("Data source not found. Please ensure `processed_school_data.csv` is correctly generated.")

elif page == "AI Advisor (Chat)":
    st.markdown('<h1 class="main-header">EduSmart AI Advisor</h1>', unsafe_allow_html=True)
    
    if use_local_llm:
        ollama_models = discover_ollama_models(ollama_models)
        st.info("Local LLM mode is enabled. The chat runs via Ollama on your machine.")
        rag_ready = st.session_state.get("local_rag") is not None
        st.caption(f"Local RAG: {'Loaded' if rag_ready else 'Not loaded'} | Index: {RAG_INDEX_PATH}")
        if df is not None and "District" in df.columns:
            district_values = sorted({str(d).strip() for d in df["District"].dropna().unique().tolist()})
            st.caption(f"Dataset districts: {', '.join(district_values[:5])} (total {len(district_values)})")
    else:
        if not api_key:
            st.warning("Local LLM is disabled (Ollama not found). Install Ollama or provide an API key to enable chat.")
    if model_choice is None:
        model_choice = ollama_model

    # Chat UI
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Container for messages
    chat_container = st.container()

    with chat_container:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    # Input area
    if prompt := st.chat_input("Ask about resource shortages, planning, or prediction results..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                try:
                    deterministic = get_deterministic_answer(prompt, df, st.session_state.get('last_prediction'))
                    if deterministic:
                        full_res = deterministic
                    elif use_local_llm:
                        with st.spinner(f"Thinking with {model_choice} (local)..."):
                            rag = st.session_state.get("local_rag")
                            if rag is None:
                                full_res = "Local knowledge base is unavailable. Please reload the app after dataset generation."
                            else:
                                retrieved = rag.retrieve(
                                    prompt,
                                    top_k=local_rag_top_k,
                                    min_score=local_rag_min_score
                                )
                                retrieved = boost_district_chunk(prompt, df, rag, retrieved)
                                if not retrieved:
                                    full_res = (
                                        "Out-of-scope for local model context. I can only answer questions grounded in "
                                        "the local EducationGPT knowledge base (processed school dataset, project report, "
                                        "and recent prediction context)."
                                    )
                                else:
                                    rag_context = format_rag_context(retrieved)
                                    system_prompt = get_advisor_system_prompt(df, st.session_state.get('last_prediction'))
                                    grounding_rules = (
                                        "Grounding policy:\n"
                                        "- Use only the retrieved local context below.\n"
                                        "- If required information is missing, reply exactly: "
                                        "'Not available in local knowledge base.'\n"
                                        "- Do not use external facts.\n"
                                        "- Keep response concise and operational."
                                    )
                                    chat_messages = [
                                        {"role": "system", "content": system_prompt},
                                        {"role": "system", "content": grounding_rules},
                                        {"role": "system", "content": f"Retrieved local context:\n{rag_context}"}
                                    ]
                                    chat_messages.extend(st.session_state.messages[-12:])
                                    full_res = query_local_llm(chat_messages, model_choice)
                                    sources = ", ".join(sorted({c.source for c in retrieved}))
                                    full_res = f"{full_res}\n\nSources: {sources}"
                    elif client:
                        with st.spinner(f"Thinking with {model_choice}..."):
                            context = get_chat_context(df, st.session_state.get('last_prediction'))
                            chat_messages = [{"role": "system", "content": context}]
                            chat_messages.extend(st.session_state.messages[-12:])

                            full_res = ""
                            try:
                                response = client.chat.completions.create(
                                    model=model_choice,
                                    messages=chat_messages,
                                    stream=True
                                )
                                for chunk in response:
                                    delta = chunk.choices[0].delta.content if chunk.choices else ""
                                    if delta:
                                        full_res += delta
                                        response_placeholder.markdown(full_res)
                            except Exception:
                                response = client.chat.completions.create(
                                    model=model_choice,
                                    messages=chat_messages,
                                    stream=False
                                )
                                full_res = response.choices[0].message.content
                    else:
                        full_res = "Chat is disabled. Install Ollama or set a valid API key to enable AI responses."
                except Exception as e:
                    full_res = f"AI Error: {e}"
                if full_res.startswith("AI Error:") and not use_local_llm and client:
                    err_msg = full_res.replace("AI Error: ", "")
                    if "404" in err_msg:
                        full_res = f"Model Error (404): The model `{model_choice}` is not available for your key or region. Try switching to another model in the sidebar."
                    elif "429" in err_msg or "Quota" in err_msg:
                        full_res = "Quota Reached: You've hit the usage limit. Please wait a minute or check your plan."
                    elif "402" in err_msg or "Insufficient Balance" in err_msg:
                        full_res = "Insufficient Balance: Your Deepseek API account has run out of credits. Please top up your balance at platform.deepseek.com."
                    else:
                        full_res = f"AI Notice: I encountered an error ({err_msg}). Here is my local analytical assessment:\n\n"
                        p = prompt.lower()
                        if "last" in p or "prediction" in p:
                            if "last_prediction" in st.session_state:
                                lp = st.session_state['last_prediction']
                                full_res += f"The school in **{lp['District']}** ({lp['LEVEL']}) is predicted as **{lp['status']}**. "
                                full_res += f"With only {lp['total_teachers']} teachers and {lp['No_of_Pakka_Class_Rooms']} classrooms, there is a measurable risk."
                            else:
                                full_res += "I don't have a recent prediction to analyze. Go to the 'Dashboard' tab and run a prediction first!"
                        elif "district" in p or "trend" in p:
                            if df is not None:
                                top = df['District'].value_counts().idxmax()
                                full_res += f"Based on the dataset, **{top}** seems to have the highest concentration of surveyed schools. "
                                full_res += f"Overall, {df['is_critical'].sum()} out of {len(df)} schools are flagged for critical shortages."
                            else:
                                full_res += "I don't have access to the dataset right now to identify trends."
                        else:
                            full_res += "I can help with school planning. Try asking about a specific district or the last prediction result."
                
                response_placeholder.markdown(full_res)
                st.session_state.messages.append({"role": "assistant", "content": full_res})
