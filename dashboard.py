import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="EduSmart AI | School Resource Dashboard",
    page_icon="🏫",
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
        --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.1);
    }

    * {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
    }

    /* Glassmorphism for containers */
    .stApp {
        background: radial-gradient(circle at top right, #1e1b4b, #0f172a);
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
        border: 1px solid var(--glass-border);
    }

    .stChatMessage[data-testimonial="user"] {
        background: rgba(99, 102, 241, 0.1);
    }

    .stChatMessage[data-testimonial="assistant"] {
        background: rgba(168, 85, 247, 0.1);
    }

    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid var(--glass-border);
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
        color: #a855f7 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration & Models ---
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
else:
    gemini_model = None

DATA_PATH = r"c:\Users\Engr.Tariq Jamal\Downloads\EMA_ML_model\processed_school_data.csv"
API_URL = "http://localhost:8000/predict"

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown('<p class="sidebar-title">🏫 EduSmart AI</p>', unsafe_allow_html=True)
    page = st.radio("Navigation", ["Dashboard & Analytics", "AI Advisor (Chat)"], label_visibility="collapsed")
    
    st.divider()
    if page == "AI Advisor (Chat)":
        st.subheader("Settings")
        model_choice = st.selectbox(
            "AI Model",
            ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro-latest"],
            index=0,
            help="Switch models if you encounter Quota or 404 errors."
        )
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.caption("v2.1.0 | Premium Edition")

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

# --- Main Page Execution ---
df = load_data()

# Re-configure Gemini based on sidebar choice
if api_key and 'model_choice' in locals():
    try:
        gemini_model = genai.GenerativeModel(model_choice)
    except:
        gemini_model = None

if page == "Dashboard & Analytics":
    st.markdown('<h1 class="main-header">School Resource Insights</h1>', unsafe_allow_html=True)
    
    if df is not None:
        # Filters in columns for cleaner dashboard
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            district_list = sorted(df['District'].unique())
            district = st.multiselect("Filter District", options=district_list, default=district_list[:3])
        with col_f2:
            level_list = sorted(df['LEVEL'].unique())
            level = st.multiselect("Filter Level", options=level_list, default=level_list)

        filtered_df = df[(df['District'].isin(district)) & (df['LEVEL'].isin(level))]

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
            
        # Prediction Form (Collapsible)
        with st.expander("🔍 Infrastructure Shortage Prediction Tool"):
            st.markdown("Enter school stats to predict if it will fall into the 'Critical' category.")
            with st.form("pred_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    gender = st.selectbox("Gender", ["Boys", "Girls", "Co-Education"])
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
                                st.error(f"⚠️ **CRITICAL SHORTAGE DETECTED** (Probability: {res['probability'][1]:.2%})")
                            else:
                                st.success(f"✅ **NORMAL STATUS** (Probability: {res['probability'][0]:.2%})")
                        else: st.error(f"API Error: {response.text}")
                    except: st.error("Connection Error: Is the API server running?")

    else:
        st.warning("⚠️ Data source not found. Please ensure `processed_school_data.csv` is correctly generated.")

elif page == "AI Advisor (Chat)":
    st.markdown('<h1 class="main-header">EduSmart AI Advisor</h1>', unsafe_allow_html=True)
    
    if not api_key:
        st.info("💡 **Developer Note:** Add your `GEMINI_API_KEY` to the `.env` file for the full LLM experience.")

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
                if gemini_model:
                    try:
                        with st.spinner(f"Thinking with {model_choice}..."):
                            context = get_chat_context(df, st.session_state.get('last_prediction'))
                            full_prompt = f"{context}\n\nUser Question: {prompt}\n\nStrategic Advice:"
                            response = gemini_model.generate_content(full_prompt)
                            full_res = response.text
                    except Exception as e:
                        err_msg = str(e)
                        if "404" in err_msg:
                            full_res = f"❌ **Model Error (404):** The model `{model_choice}` is not available for your key or region. Try switching to another model in the sidebar."
                        elif "429" in err_msg or "Quota" in err_msg:
                            full_res = f"⚠️ **Quota Reached:** You've hit the Gemini free tier limit. Please wait a minute or switch models."
                        else:
                            full_res = f"🤖 **AI Notice:** I encountered an error ({err_msg}). Here is my local analytical assessment:\n\n"
                            # Conversational Fallback
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
                else:
                    full_res = "AI Advisor is in local mode. Connect a Gemini API key to enable strategic insights."
                
                response_placeholder.markdown(full_res)
                st.session_state.messages.append({"role": "assistant", "content": full_res})



