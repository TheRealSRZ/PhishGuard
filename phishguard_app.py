import os
import platform
import io
import string
import random
import time
import re
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px

# Scikit-learn & NLTK
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC  # <--- FIX: Added LinearSVC here!
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Selenium for web scraping
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# --- Ensure NLTK Data is available ---
@st.cache_resource
def download_nltk_data():
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords', quiet=True)

download_nltk_data()

# --- Page Config ---
st.set_page_config(page_title="PhishGuard AI", layout="wide", page_icon="🛡️", initial_sidebar_state="expanded")

# --- Custom CSS for visual polish ---
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E88E5; margin-bottom: 0px; }
    .sub-header { font-size: 1.2rem; color: #555; margin-bottom: 30px; }
    .stAlert { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- Directory Setup ---
DATASETS_DIR = "datasets"
JOBLIBS_DIR = "joblibs"
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(JOBLIBS_DIR, exist_ok=True)

# --- Session State Management ---
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'flow_step' not in st.session_state:
    st.session_state.flow_step = 0
if 'log_tab1' not in st.session_state:
    st.session_state.log_tab1 = ""
if 'log_tab2' not in st.session_state:
    st.session_state.log_tab2 = ""
if 'training_metrics' not in st.session_state:
    st.session_state.training_metrics = []

# =======================================================================
# SIDEBAR: VISUAL FLOW TRACKER
# =======================================================================
with st.sidebar:
    # 1. Use columns to squeeze the image into the exact center
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        # Replace "logo.png" with the actual name of your saved logo file
        st.image("logo/logo_clear.png", width=120)
        
    # 2. Use a touch of HTML to center the header text
    st.markdown("<h2 style='text-align: center; margin-top: -10px;'>System Status</h2>", unsafe_allow_html=True)
    st.divider()
    
    steps = [
        "1. Data Gathering",
        "2. NLP Preprocessing",
        "3. Model Training",
        "4. System Ready"
    ]

    for i, step in enumerate(steps):
        if st.session_state.flow_step > i:
            st.success(f"✅ **{step}**")
        elif st.session_state.flow_step == i:
            st.info(f"🔄 **{step}** (Active)")
        else:
            st.markdown(f"<div style='padding:10px; border-radius:5px; background-color:rgba(128,128,128,0.1); color:gray; margin-bottom:10px;'>⏳ {step}</div>", unsafe_allow_html=True)
            
    st.divider()
    st.markdown("<p style='text-align: center; color: gray; font-size: 0.85rem;'>Developed for threat intelligence & phishing mitigation.</p>", unsafe_allow_html=True)

# =======================================================================
# MAIN HEADER
# =======================================================================
st.markdown('<p class="main-header">🛡️ PhishGuard AI Scanner</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Natural Language Processing for Threat Detection</p>', unsafe_allow_html=True)

# =======================================================================
# MAIN UI TABS
# =======================================================================
tab1, tab2, tab3, tab4 = st.tabs(["🌐 Data Gathering", "🧠 AI Training", "🧪 Sampler Sandbox", "🛡️ Live Detection"])

csv_files = [f for f in os.listdir(DATASETS_DIR) if f.endswith('.csv')]
pkl_files = [f for f in os.listdir(JOBLIBS_DIR) if f.endswith('.pkl')]

# -----------------------------------------------------------------------
# TAB 1: DATA GATHERING
# -----------------------------------------------------------------------
with tab1:
    st.markdown("### 📥 Build Your Dataset")
    st.write("Enter target URLs or local `.csv` files below to crawl, extract, and clean text data into a unified dataset.")
    
    default_urls = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv\nphishing_messages.csv"
    targets_input = st.text_area("Target Sources (One per line):", value=default_urls, height=120)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        out_filename = st.text_input("Master Output Filename:", value="dataset.csv", help="Saved to the datasets/ folder")
    with col2:
        st.write("") 
        st.write("")
        start_crawl = st.button("🚀 Start Extraction", use_container_width=True, type="primary")

    if start_crawl:
        st.session_state.flow_step = 0
        inputs = [i.strip() for i in targets_input.split('\n') if i.strip()]
        
        if not inputs:
            st.warning("⚠️ Please provide at least one input source.")
        else:
            st.session_state.log_tab1 = "--- Initializing Extraction Protocol ---\n"
            st.session_state.log_tab1 += f"Environment: {platform.system()} OS Detected. Configuring drivers...\n"
            
            with st.status("Crawling & Processing Data...", expanded=True) as status:
                log_placeholder_1 = st.empty()
                log_placeholder_1.code(st.session_state.log_tab1, language="bash")
                
                options = Options()
                options.add_argument("--headless")
                options.add_argument("--disable-gpu")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--log-level=3")

                if platform.system() == "Linux":
                    options.binary_location = "/usr/bin/chromium"
                    svc = Service("/usr/bin/chromedriver")
                else:
                    svc = Service(ChromeDriverManager().install())

                driver = None
                master_df = pd.DataFrame(columns=['label', 'text_content'])

                with st.spinner("Processing datasets..."):
                    for item in inputs:
                        df = None
                        st.session_state.log_tab1 += f"\n[Target] {item}\n"
                        log_placeholder_1.code(st.session_state.log_tab1, language="bash")
                        
                        try:
                            if item.startswith('http'):
                                if driver is None:
                                    st.session_state.log_tab1 += "  -> [System] Spinning up headless Chrome Webdriver...\n"
                                    driver = webdriver.Chrome(service=svc, options=options)
                                
                                st.session_state.log_tab1 += "  -> [Network] Initiating HTTP GET request...\n"
                                log_placeholder_1.code(st.session_state.log_tab1, language="bash")
                                
                                driver.get(item)
                                page_text = driver.find_element(By.TAG_NAME, "body").get_attribute("textContent")
                                
                                st.session_state.log_tab1 += f"  -> [Network] Downloaded {len(page_text):,} characters of raw content.\n"
                                st.session_state.log_tab1 += "  -> [Parser] Converting raw text into tabular format...\n"
                                log_placeholder_1.code(st.session_state.log_tab1, language="bash")
                                
                                if '.tsv' in item.lower():
                                    df = pd.read_csv(io.StringIO(page_text), sep='\t', header=None, on_bad_lines='skip')
                                else:
                                    df = pd.read_csv(io.StringIO(page_text), on_bad_lines='skip')
                            else:
                                st.session_state.log_tab1 += "  -> [Local] Scanning local file paths...\n"
                                file_path = item if os.path.exists(item) else os.path.join(DATASETS_DIR, item)
                                if os.path.exists(file_path):
                                    st.session_state.log_tab1 += f"  -> [Local] File located at {file_path}. Reading data...\n"
                                    if file_path.lower().endswith('.tsv'):
                                        df = pd.read_csv(file_path, sep='\t', header=None, on_bad_lines='skip')
                                    else:
                                        df = pd.read_csv(file_path, on_bad_lines='skip')
                                else:
                                    st.session_state.log_tab1 += f"  ❌ Error: File not found.\n"
                                    log_placeholder_1.code(st.session_state.log_tab1, language="bash")
                                    continue

                            st.session_state.log_tab1 += f"  -> [Data] Raw dimensions: {df.shape[0]:,} rows, {df.shape[1]} columns.\n"
                            st.session_state.log_tab1 += "  -> [Data] Initiating Smart Column Detection algorithm...\n"
                            log_placeholder_1.code(st.session_state.log_tab1, language="bash")

                            # Smart Cleanup
                            df.dropna(axis=1, how='all', inplace=True)
                            
                            if 'phishing_messages.csv' in item:
                                # 1. Set the text column to match Column D in your file
                                text_col = 'message' 
                                
                                # 2. Force every row in this file to be labeled as phishing ('1')
                                df['manual_label'] = '1'
                                label_col = 'manual_label'
                            else:
                                # Fallback to smart detection for other files
                                lengths = {col: df[col].astype(str).str.len().mean() for col in df.columns}
                                text_col = max(lengths, key=lengths.get)
                                remaining_cols = [col for col in df.columns if col != text_col]
                                if remaining_cols:
                                    unique_counts = {c: df[c].nunique() for c in remaining_cols}
                                    label_col = min(unique_counts, key=unique_counts.get)
                            remaining_cols = [col for col in df.columns if col != text_col]
                            if not remaining_cols: 
                                st.session_state.log_tab1 += "  ❌ Error: Could not isolate labels from text.\n"
                                continue
                                
                            unique_counts = {c: df[c].nunique() for c in remaining_cols}
                            label_col = min(unique_counts, key=unique_counts.get)

                            st.session_state.log_tab1 += f"  -> [Data] Identified '{label_col}' as Label column and '{text_col}' as Text block.\n"
                            st.session_state.log_tab1 += "  -> [Data] Normalizing labels (Mapping phishing/spam to 1, benign to 0)...\n"
                            log_placeholder_1.code(st.session_state.log_tab1, language="bash")

                            df = df[[label_col, text_col]]
                            df.columns = ['raw_label', 'text_content']
                            phish_keywords = ['spam', '1', '1.0', 'phishing', 'malicious', 'bad']
                            df['label'] = df['raw_label'].apply(lambda x: 1 if str(x).strip().lower() in phish_keywords else 0)
                            
                            initial_count = len(df)
                            df = df[['label', 'text_content']].dropna().drop_duplicates()
                            dropped_count = initial_count - len(df)
                            
                            st.session_state.log_tab1 += f"  -> [Data] Dropped {dropped_count:,} blank or duplicate rows.\n"
                            st.session_state.log_tab1 += f"  ✅ Success: Extracted {len(df):,} clean records.\n"
                            master_df = pd.concat([master_df, df], ignore_index=True)

                        except Exception as e:
                            st.session_state.log_tab1 += f"  ❌ Error: {str(e)[:100]}...\n"
                        
                        log_placeholder_1.code(st.session_state.log_tab1, language="bash")

                if driver: driver.quit()

                if not master_df.empty:
                    st.session_state.log_tab1 += "\n--- Finalizing Master Dataset ---\n"
                    st.session_state.log_tab1 += "  -> Verifying cross-dataset duplicates...\n"
                    
                    final_initial = len(master_df)
                    master_df.drop_duplicates(subset=['text_content'], inplace=True)
                    st.session_state.log_tab1 += f"  -> Removed {final_initial - len(master_df):,} cross-dataset duplicates.\n"
                    
                    save_path = os.path.join(DATASETS_DIR, out_filename)
                    master_df.to_csv(save_path, index=False)
                    st.session_state.log_tab1 += f"\n🎉 DONE! Saved total {len(master_df):,} unique records to {save_path}"
                    
                    log_placeholder_1.code(st.session_state.log_tab1, language="bash")
                    st.session_state.flow_step = 1
                    status.update(label="Extraction Complete!", state="complete", expanded=False)
                    time.sleep(1)
                    st.rerun()
                else:
                    status.update(label="Extraction Failed", state="error", expanded=True)

    elif st.session_state.log_tab1:
        with st.expander("Show Last Extraction Logs", expanded=False):
            st.code(st.session_state.log_tab1, language="bash")

# -----------------------------------------------------------------------
# TAB 2: AI MODEL TRAINING
# -----------------------------------------------------------------------
with tab2:
    st.markdown("### ⚙️ AI Model Training")
    
    colA, colB = st.columns(2)
    with colA:
        with st.container(border=True):
            st.markdown("#### Option A: Train New Models")
            selected_csv = st.selectbox("Select CSV Dataset:", ["No CSV found"] if not csv_files else csv_files)
            train_btn = st.button("🧠 Execute Training", type="primary", use_container_width=True)
        
    with colB:
        with st.container(border=True):
            st.markdown("#### Option B: Load Pre-Trained")
            selected_pkl = st.selectbox("Select PKL Model:", ["No saved models found"] if not pkl_files else pkl_files)
            load_btn = st.button("⚡ Load Weights", use_container_width=True)

    if load_btn and selected_pkl != "No saved models found":
            target_pkl_path = os.path.join(JOBLIBS_DIR, selected_pkl)
            with st.spinner("Unpacking AI weights..."):
                try:
                    saved_data = joblib.load(target_pkl_path)
                    st.session_state.vectorizer = saved_data['vectorizer']
                    st.session_state.models = saved_data['models']
                    st.session_state.training_metrics = saved_data.get('metrics', [])
                    
                    st.session_state.flow_step = 4
                    st.toast("✅ Models loaded successfully!", icon="🛡️")
                except Exception as e:
                    st.error(f"Failed to load: {e}")

    elif train_btn and selected_csv != "No CSV found":
        target_csv_path = os.path.join(DATASETS_DIR, selected_csv)
        st.session_state.log_tab2 = f"--- INITIATING NLP ---\n"
        st.session_state.log_tab2 += f"> Target Architecture: {target_csv_path}\n"
        
        with st.status("Executing NLP...", expanded=True) as status:
            log_placeholder_2 = st.empty()
            
            try:
                df = pd.read_csv(target_csv_path)
                st.session_state.flow_step = 1
                st.session_state.log_tab2 += f"> Dataset loaded into memory. Total rows: {len(df):,}.\n\n"
                
                st.session_state.log_tab2 += "[Step 1] Executing Natural Language Preprocessing...\n"
                st.session_state.log_tab2 += "  -> Stripping punctuation and converting string formats to lowercase.\n"
                st.session_state.log_tab2 += "  -> Querying NLTK Dictionary for English stopword exclusion list...\n"
                log_placeholder_2.code(st.session_state.log_tab2, language="bash")

                def clean_text(text):
                    try:
                        text = str(text).lower().translate(str.maketrans('', '', string.punctuation))
                        stops = set(stopwords.words('english'))
                        return " ".join([w for w in text.split() if w not in stops])
                    except: return ""

                df['cleaned_text'] = df['text_content'].apply(clean_text)
                
                initial_len = len(df)
                df.dropna(subset=['cleaned_text'], inplace=True)
                st.session_state.log_tab2 += f"  -> Removed {initial_len - len(df):,} corrupted/empty text bodies.\n"

                st.session_state.log_tab2 += "\n[Step 2] Feature Extraction (TF-IDF Vectorization)...\n"
                st.session_state.log_tab2 += "  -> Calculating Term Frequency-Inverse Document Frequency.\n"
                log_placeholder_2.code(st.session_state.log_tab2, language="bash")
                
                st.session_state.vectorizer = TfidfVectorizer(max_features=5000)
                X_vec = st.session_state.vectorizer.fit_transform(df['cleaned_text'])
                
                st.session_state.log_tab2 += f"  -> Successfully generated numerical vocabulary. Matrix shape: {X_vec.shape[0]:,} rows x {X_vec.shape[1]:,} features.\n"
                
                X_train, X_test, y_train, y_test = train_test_split(X_vec, df['label'], test_size=0.2, random_state=42)
                st.session_state.log_tab2 += f"  -> Shuffling and splitting data arrays: 80% Training ({X_train.shape[0]:,} samples), 20% Testing ({X_test.shape[0]:,} samples).\n"
                st.session_state.flow_step = 2
                log_placeholder_2.code(st.session_state.log_tab2, language="bash")

                models_to_train = {
                                    "Naïve Bayes": MultinomialNB(),
                                    "SVM (Linear)": CalibratedClassifierCV(LinearSVC(random_state=42)), 
                                    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
                                }
                st.session_state.models.clear()
                st.session_state.training_metrics = []

                st.session_state.log_tab2 += "\n[Step 3] AI Model Compilation & Calibration...\n"

                for name, model in models_to_train.items():
                    st.session_state.log_tab2 += f"\n  --- Engine: {name} ---\n"
                    st.session_state.log_tab2 += f"      -> Fitting internal weights using {X_train.shape[0]:,} training vectors...\n"
                    log_placeholder_2.code(st.session_state.log_tab2, language="bash")
                    
                    model.fit(X_train, y_train)
                    
                    st.session_state.log_tab2 += f"      -> Evaluating performance against {X_test.shape[0]:,} isolated test vectors...\n"
                    log_placeholder_2.code(st.session_state.log_tab2, language="bash")
                    
                    preds = model.predict(X_test)
                    
                    st.session_state.training_metrics.append({
                        "Model": name,
                        "Accuracy": float(accuracy_score(y_test, preds) * 100),
                        "Precision": float(precision_score(y_test, preds, zero_division=0) * 100),
                        "Recall": float(recall_score(y_test, preds, zero_division=0) * 100),
                        "F1-Score": float(f1_score(y_test, preds, zero_division=0) * 100)
                    })
                    st.session_state.models[name] = model
                    st.session_state.log_tab2 += "      -> Calibration complete.\n"

                save_filename = f"phishguard_models_{len(df)}_records.pkl"
                save_path = os.path.join(JOBLIBS_DIR, save_filename)
                
                st.session_state.log_tab2 += "\n[Step 4] Serialization\n"
                st.session_state.log_tab2 += f"  -> Compiling Vectorizer, AI Models, and Metrics into a single binary .pkl bundle...\n"
                joblib.dump({
                    'vectorizer': st.session_state.vectorizer, 
                    'models': st.session_state.models,
                    'metrics': st.session_state.training_metrics 
                }, save_path)
                
                st.session_state.log_tab2 += f"\n✅ TRAINING COMPLETE. Data saved safely to '{save_filename}'."
                log_placeholder_2.code(st.session_state.log_tab2, language="bash")
                
                st.session_state.flow_step = 4
                status.update(label="Training Complete!", state="complete", expanded=False)
                st.toast("✅ Training Complete!", icon="🚀")

            except Exception as e:
                status.update(label="Failed", state="error")
                st.error(f"Error: {e}")

    # Display Metrics if they exist
    if st.session_state.training_metrics:
        st.markdown("### 📊 Engine Performance Comparison")
        
        # Convert metrics to a DataFrame
        metrics_df = pd.DataFrame(st.session_state.training_metrics)
        
        # 1. 'Melt' the dataframe so Plotly knows how to group the bars
        df_melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        
        # 2. Create a visually striking grouped bar chart
        fig = px.bar(
            df_melted, 
            x="Model", 
            y="Score", 
            color="Metric", 
            barmode="group",
            text_auto='.1f' # Displays the exact percentage on top of each bar
        )
        
        # 3. Lock the Y-axis to 100% so it's visually accurate
        fig.update_layout(
            yaxis_title="Score (%)",
            xaxis_title="AI Engine",
            yaxis_range=[0, 100], 
            margin=dict(t=20, b=20)
        )
        
        # 4. Render the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Optional: Keep a clean dataframe view hidden in an expander
        with st.expander("🔍 View Exact Percentage Metrics"):
                    st.dataframe(
                        metrics_df.style.format({
                            "Accuracy": "{:.2f}%",
                            "Precision": "{:.2f}%",
                            "Recall": "{:.2f}%",
                            "F1-Score": "{:.2f}%"
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # --- NEW: Downloadable Audit Log ---
                    csv_export = metrics_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Audit Log (CSV)",
                        data=csv_export,
                        file_name=f"phishguard_audit_log.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    # -----------------------------------

    if st.session_state.log_tab2:
        with st.expander("View Raw Compilation Logs"):
            st.code(st.session_state.log_tab2, language="bash")

# -----------------------------------------------------------------------
# TAB 3: DATA SAMPLER
# -----------------------------------------------------------------------
with tab3:
    st.markdown("### 🧪 Synthetic Generation & Analysis")
    st.caption("Extract real samples or use a Markov Chain to generate synthetic threats.")
    
    sampler_csv = st.selectbox("Select Dataset:", ["No CSV found"] if not csv_files else csv_files, key="sampler_box")
    
    def load_sampler_df(filename):
        if filename == "No CSV found": return None
        try:
            df = pd.read_csv(os.path.join(DATASETS_DIR, filename))
            if not {'text_content', 'label'}.issubset(df.columns): return None
            return df
        except: return None

    # --- NEW: Visual Dataset Overview ---
    if sampler_csv != "No CSV found":
        df_sampler = load_sampler_df(sampler_csv)
        if df_sampler is not None:
            with st.expander("📊 View Dataset Composition", expanded=True):
                # Count the labels and map them to readable names
                counts = df_sampler['label'].value_counts().reset_index()
                counts.columns = ['Label', 'Count']
                counts['Label Name'] = counts['Label'].map({1: 'Phishing (Malicious)', 0: 'Ham (Safe)'})
                
                # Create a sleek donut chart
                pie_fig = px.pie(
                    counts, 
                    values='Count', 
                    names='Label Name', 
                    color='Label Name',
                    color_discrete_map={'Phishing (Malicious)': '#ff4b4b', 'Ham (Safe)': '#00cc96'},
                    hole=0.4 # Makes it a donut chart!
                )
                pie_fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
                st.plotly_chart(pie_fig, use_container_width=True)
    st.divider()
    # ------------------------------------
    
    c1, c2, c3 = st.columns(3)
    extract_phish = c1.button("🔴 Extract Real Phishing", use_container_width=True)
    extract_safe = c2.button("🟢 Extract Safe (Ham)", use_container_width=True)
    extract_synth = c3.button("🤖 Generate Synthetic", use_container_width=True)

    if extract_phish or extract_safe:
        df = load_sampler_df(sampler_csv)
        if df is not None:
            label_type = 1 if extract_phish else 0
            filtered_df = df[df['label'] == label_type]
            if not filtered_df.empty:
                samples = filtered_df.sample(n=min(3, len(filtered_df)))['text_content'].tolist()
                st.markdown(f"#### {'🔴 Real Phishing Threats' if label_type == 1 else '🟢 Legitimate Communications'}")
                for text in samples:
                    st.info(f"❝ {text} ❞")
            else:
                st.warning("No matching records found.")

    if extract_synth:
        df = load_sampler_df(sampler_csv)
        if df is not None:
            phishing_texts = df[df['label'] == 1]['text_content'].dropna().tolist()
            if phishing_texts:
                with st.spinner("AI is dreaming up threats..."):
                    markov_dict = {}
                    for text in phishing_texts:
                        words = str(text).split()
                        for i in range(len(words) - 1):
                            current_word, next_word = words[i], words[i + 1]
                            if current_word not in markov_dict: markov_dict[current_word] = []
                            markov_dict[current_word].append(next_word)

                    st.markdown("#### 🤖 Synthetic AI Phishing Generates")
                    for _ in range(3):
                        if not markov_dict: break
                        caps = [w for w in markov_dict.keys() if w[0].isupper()]
                        curr = random.choice(caps) if caps else random.choice(list(markov_dict.keys()))
                        sentence = [curr]
                        for _ in range(20):
                            if curr in markov_dict and markov_dict[curr]:
                                nxt = random.choice(markov_dict[curr])
                                sentence.append(nxt)
                                curr = nxt
                            else: break
                        st.error(f"❝ {' '.join(sentence)}... ❞")

# -----------------------------------------------------------------------
# TAB 4: LIVE DETECTION
# -----------------------------------------------------------------------
with tab4:
    st.markdown("### 🛡️ Threat Scanner")

    if not st.session_state.models:
        st.warning("⚠️ Action Required: Please train or load AI models in Tab 2 first.", icon="🛑")
    else:
        active_model_name = st.selectbox("Select Active Engine:", list(st.session_state.models.keys()))
        st.divider()
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### Option 1: Text Analysis")
            raw_text_input = st.text_area("Paste suspicious content here:", height=150, placeholder="Dear customer, your account has been suspended...")
            analyze_text_btn = st.button("🔍 Scan Text", type="primary", use_container_width=True)
            
        with col_right:
            st.markdown("#### Option 2: Live URL Scrape")
            url_input = st.text_input("Enter URL:", placeholder="https://suspicious-link.com")
            st.caption("Note: Cloud deployment requires Chromium packages to scrape.")
            analyze_url_btn = st.button("🌐 Scrape & Scan", use_container_width=True)

        st.markdown("#### ⚙️ Security Operations")
        user_threshold = st.slider(
            "Set AI Confidence Threshold (Higher = Less Strict, Lower = More Strict)", 
            min_value=50, max_value=99, value=75, step=1
        )
        st.divider()

        def execute_prediction(raw_text):
            if not raw_text.strip():
                st.warning("Input is empty.")
                return

            model = st.session_state.models[active_model_name]
            text_lower = raw_text.lower().translate(str.maketrans('', '', string.punctuation))
            stops = set(stopwords.words('english'))
            cleaned = " ".join([w for w in text_lower.split() if w not in stops])
            
            vectorized_text = st.session_state.vectorizer.transform([cleaned])

            probabilities = model.predict_proba(vectorized_text)[0]
            phishing_confidence = probabilities[1] * 100 # Percentage chance it is phishing
            safe_confidence = probabilities[0] * 100     # Percentage chance it is safe

            feature_names = st.session_state.vectorizer.get_feature_names_out()
            tfidf_scores = vectorized_text.toarray()[0]
            top_indices = tfidf_scores.argsort()[-8:][::-1]
            flagged_words = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]

            # We compare the AI's confidence against the user's slider
            if phishing_confidence >= user_threshold:
                st.error(f"### 🚨 CRITICAL ALERT: Phishing Detected!")
                st.progress(int(phishing_confidence), text=f"Threat Confidence: {phishing_confidence:.1f}%")
            elif phishing_confidence >= 40:
                # A new "middle ground" for suspicious but not outright malicious text
                st.warning(f"### ⚠️ SUSPICIOUS: Proceed with caution.")
                st.progress(int(phishing_confidence), text=f"Threat Confidence: {phishing_confidence:.1f}%")
            else:
                st.success(f"### ✅ SAFE: No obvious malicious intent.")
                st.progress(int(safe_confidence), text=f"Safety Confidence: {safe_confidence:.1f}%")

            with st.expander("View Scanned Content"):
                display_text = raw_text[:2000] # Capture a good chunk of text
                
                # Fixed: Swapped 'prediction == 1' for 'phishing_confidence >= 40'
                if phishing_confidence >= 40 and flagged_words:
                    st.markdown(f"**🚩 Flagged Keywords:** `{', '.join(flagged_words)}`")
                    st.divider()
                    
                    # Highlight the flagged words in the text using HTML
                    for word in flagged_words:
                        # Regex ensures we only highlight whole words, case-insensitive
                        pattern = re.compile(rf'\b({re.escape(word)})\b', re.IGNORECASE)
                        display_text = pattern.sub(
                            r'<mark style="background-color: #ffcccc; color: #990000; font-weight: bold; padding: 2px; border-radius: 3px;">\1</mark>', 
                            display_text
                        )
                
                # Render the text with HTML enabled to show the <mark> highlights
                st.markdown(display_text + ("..." if len(raw_text) > 2000 else ""), unsafe_allow_html=True)

        if analyze_text_btn:
            execute_prediction(raw_text_input)

        if analyze_url_btn:
            if not url_input.startswith("http"):
                st.warning("Invalid URL. Include http:// or https://")
            else:
                with st.spinner("Scraping target..."):
                    try:
                        options = Options()
                        options.add_argument("--headless")
                        options.add_argument("--disable-gpu")
                        options.add_argument("--no-sandbox")
                        options.add_argument("--disable-dev-shm-usage")
                        
                        # --- NEW OS-AWARE SELENIUM SETUP ---
                        if platform.system() == "Linux":
                            options.binary_location = "/usr/bin/chromium"
                            svc = Service("/usr/bin/chromedriver")
                        else:
                            svc = Service(ChromeDriverManager().install())
                            
                        driver = webdriver.Chrome(service=svc, options=options)
                        # -----------------------------------
                        
                        driver.get(url_input)
                        time.sleep(2)
                        page_text = driver.find_element(By.TAG_NAME, "body").text
                        driver.quit()
                        
                        execute_prediction(page_text[:5000])
                        
                    except Exception as e:
                        st.error(f"⚠️ Scraping Failed. If on Streamlit Cloud, ensure packages.txt is configured. Error details: {e}")