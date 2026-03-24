import os
import io
import string
import random
import time
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Scikit-learn & NLTK
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Selenium for the web scraping stuff
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
st.set_page_config(page_title="PhishGuard AI", layout="wide", page_icon="🛡️")

# --- Directory Setup ---
DATASETS_DIR = "datasets"
JOBLIBS_DIR = "joblibs"

# Ensure the folders exist so the app doesn't crash on first run
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

# =======================================================================
# VISUAL FLOW TRACKER
# =======================================================================
st.markdown("### 🚦 Live System Flow")
flow_cols = st.columns(4)
steps = [
    "1. Data Gathering",
    "2. NLP Preprocessing",
    "3. Model Training",
    "4. System Ready"
]

for i, col in enumerate(flow_cols):
    if st.session_state.flow_step > i:
        col.success(f"✅ {steps[i]}")
    elif st.session_state.flow_step == i:
        col.info(f"🔄 {steps[i]} (Active)")
    else:
        col.markdown(f"<div style='padding:15px; border-radius:5px; background-color:#1E1E1E; color:#555; text-align:center; font-weight:bold;'>{steps[i]}</div>", unsafe_allow_html=True)

st.divider()

# =======================================================================
# MAIN UI TABS
# =======================================================================
tab1, tab2, tab3, tab4 = st.tabs(["1. Data Gathering", "2. AI Model Training", "3. Data Sampler", "4. Live Detection"])

# Helper function to get CSVs and PKLs from their respective folders
csv_files = [f for f in os.listdir(DATASETS_DIR) if f.endswith('.csv')]
pkl_files = [f for f in os.listdir(JOBLIBS_DIR) if f.endswith('.pkl')]

# -----------------------------------------------------------------------
# TAB 1: DATA GATHERING
# -----------------------------------------------------------------------
with tab1:
    st.header("🌐 Data Gathering")
    st.write("Target URLs or Local CSV Files (One per line):")
    
    default_urls = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv\nphishing_messages.csv"
    targets_input = st.text_area("Inputs", value=default_urls, height=100, label_visibility="collapsed")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        out_filename = st.text_input("Output Master Filename:", value="combined_raw_dataset.csv")
    with col2:
        st.write("") # Spacing
        st.write("")
        start_crawl = st.button("Start Processing", use_container_width=True)

    log_placeholder_1 = st.empty()

    if start_crawl:
        st.session_state.flow_step = 0
        inputs = [i.strip() for i in targets_input.split('\n') if i.strip()]
        
        if not inputs:
            st.error("No inputs provided.")
        else:
            st.session_state.log_tab1 = "--- Initializing Data Gathering ---\n"
            log_placeholder_1.text_area("Gathering Logs", st.session_state.log_tab1, height=300)
            
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            options.add_argument("--log-level=3")
            driver = None
            master_df = pd.DataFrame(columns=['label', 'text_content'])

            with st.spinner("Processing datasets..."):
                for item in inputs:
                    df = None
                    st.session_state.log_tab1 += f"\n[Processing] {item}\n"
                    log_placeholder_1.text_area("Gathering Logs", st.session_state.log_tab1, height=300)
                    
                    try:
                        if item.startswith('http'):
                            if driver is None:
                                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
                            driver.get(item)
                            page_text = driver.find_element(By.TAG_NAME, "body").get_attribute("textContent")
                            if '.tsv' in item.lower():
                                df = pd.read_csv(io.StringIO(page_text), sep='\t', header=None, on_bad_lines='skip')
                            else:
                                df = pd.read_csv(io.StringIO(page_text), on_bad_lines='skip')
                        else:
                            # If they provided a local path, check if it's already in the datasets folder
                            file_path = item if os.path.exists(item) else os.path.join(DATASETS_DIR, item)
                            if os.path.exists(file_path):
                                if file_path.lower().endswith('.tsv'):
                                    df = pd.read_csv(file_path, sep='\t', header=None, on_bad_lines='skip')
                                else:
                                    df = pd.read_csv(file_path, on_bad_lines='skip')
                            else:
                                st.session_state.log_tab1 += f" -> [ERROR]: Local file '{item}' not found.\n"
                                log_placeholder_1.text_area("Gathering Logs", st.session_state.log_tab1, height=300)
                                continue

                        # Smart Column Detection
                        df.dropna(axis=1, how='all', inplace=True)
                        lengths = {col: df[col].astype(str).str.len().mean() for col in df.columns}
                        text_col = max(lengths, key=lengths.get)
                        
                        remaining_cols = [col for col in df.columns if col != text_col]
                        if not remaining_cols: continue
                        
                        unique_counts = {col: df[col].nunique() for col in remaining_cols}
                        label_col = min(unique_counts, key=unique_counts.get)

                        df = df[[label_col, text_col]]
                        df.columns = ['raw_label', 'text_content']
                        
                        phish_keywords = ['spam', '1', '1.0', 'phishing', 'malicious', 'bad']
                        df['label'] = df['raw_label'].apply(lambda x: 1 if str(x).strip().lower() in phish_keywords else 0)
                        
                        df = df[['label', 'text_content']]
                        df.dropna(subset=['text_content'], inplace=True)
                        df.drop_duplicates(subset=['text_content'], inplace=True)

                        st.session_state.log_tab1 += f" -> Success: Cleaned & Extracted {len(df)} records.\n"
                        master_df = pd.concat([master_df, df], ignore_index=True)

                    except Exception as e:
                        st.session_state.log_tab1 += f" -> [ERROR]: {e}\n"
                    
                    log_placeholder_1.text_area("Gathering Logs", st.session_state.log_tab1, height=300)

                if driver:
                    driver.quit()

                if not master_df.empty:
                    master_df.drop_duplicates(subset=['text_content'], inplace=True)
                    # Save output to datasets directory
                    save_path = os.path.join(DATASETS_DIR, out_filename)
                    master_df.to_csv(save_path, index=False)
                    st.session_state.log_tab1 += f"\nSUCCESS! Saved a total of {len(master_df)} unique records to {save_path}\n"
                    st.session_state.flow_step = 1
                    st.rerun() # Refresh file lists
                else:
                    st.session_state.log_tab1 += "\nFAILED: No data retrieved across all inputs.\n"
                    
                log_placeholder_1.text_area("Gathering Logs", st.session_state.log_tab1, height=300)

    elif st.session_state.log_tab1:
        log_placeholder_1.text_area("Gathering Logs", st.session_state.log_tab1, height=300)

# -----------------------------------------------------------------------
# TAB 2: AI MODEL TRAINING
# -----------------------------------------------------------------------
with tab2:
    st.header("🧠 AI Model Training")
    
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("Option A: Train New Models")
        selected_csv = st.selectbox("Select CSV Dataset:", ["No CSV found"] if not csv_files else csv_files)
        train_btn = st.button("Train & Save AI Models", type="primary")
        
    with colB:
        st.subheader("Option B: Quick-Load Pre-Trained")
        selected_pkl = st.selectbox("Select PKL Model:", ["No saved models found"] if not pkl_files else pkl_files)
        load_btn = st.button("⚡ Load Saved Models")

    log_placeholder_2 = st.empty()

    # --- Load Logic ---
    if load_btn and selected_pkl != "No saved models found":
        target_pkl_path = os.path.join(JOBLIBS_DIR, selected_pkl)
        st.session_state.log_tab2 = f"--- QUICK LOAD INITIATED ---\n>>> Unpacking AI Brains from: {target_pkl_path}...\n"
        log_placeholder_2.text_area("Training Logs", st.session_state.log_tab2, height=350)
        
        with st.spinner("Loading models into memory..."):
            time.sleep(0.5)
            try:
                saved_data = joblib.load(target_pkl_path)
                st.session_state.vectorizer = saved_data['vectorizer']
                st.session_state.models = saved_data['models']
                
                st.session_state.log_tab2 += f"   ✓ Restored TF-IDF Vocabulary ({len(st.session_state.vectorizer.get_feature_names_out())} terms)\n"
                st.session_state.log_tab2 += f"   ✓ Restored {len(st.session_state.models)} AI Engines\n"
                st.session_state.log_tab2 += "\n✅ SYSTEM READY. Proceed to Tab 4 (Live Detection)."
                st.session_state.flow_step = 4
                log_placeholder_2.text_area("Training Logs", st.session_state.log_tab2, height=350)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load models: {e}")

    # --- Train Logic ---
    elif train_btn and selected_csv != "No CSV found":
        target_csv_path = os.path.join(DATASETS_DIR, selected_csv)
        st.session_state.log_tab2 = f">>> Loading dataset: {target_csv_path}...\n"
        log_placeholder_2.text_area("Training Logs", st.session_state.log_tab2, height=350)
        
        with st.spinner("Executing NLP Pipeline & Training Models..."):
            try:
                df = pd.read_csv(target_csv_path)
                st.session_state.log_tab2 += f"[1/5] Loaded {len(df)} records.\n"
                st.session_state.flow_step = 1
                
                st.session_state.log_tab2 += "\n[2/5] Preprocessing NLP data...\n"
                log_placeholder_2.text_area("Training Logs", st.session_state.log_tab2, height=350)

                def clean_text(text):
                    try:
                        text = str(text).lower().translate(str.maketrans('', '', string.punctuation))
                        stops = set(stopwords.words('english'))
                        return " ".join([w for w in text.split() if w not in stops])
                    except: return ""

                df['cleaned_text'] = df['text_content'].apply(clean_text)
                df.replace('', np.nan, inplace=True)
                df.dropna(subset=['cleaned_text'], inplace=True)

                st.session_state.log_tab2 += "[3/5] Extracting TF-IDF Features & Splitting Data...\n"
                log_placeholder_2.text_area("Training Logs", st.session_state.log_tab2, height=350)
                
                st.session_state.vectorizer = TfidfVectorizer(max_features=5000)
                X_vec = st.session_state.vectorizer.fit_transform(df['cleaned_text'])
                X_train, X_test, y_train, y_test = train_test_split(X_vec, df['label'], test_size=0.2, random_state=42)

                st.session_state.flow_step = 2
                
                st.session_state.log_tab2 += "[4/5] Training the 3 AI Models...\n"
                models_to_train = {
                    "Naïve Bayes (Probabilistic)": MultinomialNB(),
                    "SVM (Linear/Geometric)": SVC(kernel='linear', random_state=42),
                    "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=100, random_state=42)
                }

                st.session_state.models.clear()
                results = []

                for name, model in models_to_train.items():
                    st.session_state.log_tab2 += f"   -> Compiling engine: {name}...\n"
                    log_placeholder_2.text_area("Training Logs", st.session_state.log_tab2, height=350)
                    
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    
                    acc = accuracy_score(y_test, preds) * 100
                    prec = precision_score(y_test, preds, zero_division=0) * 100
                    rec = recall_score(y_test, preds, zero_division=0) * 100
                    f1 = f1_score(y_test, preds, zero_division=0) * 100
                    
                    results.append(f"{name.split(' ')[0]:<15} | {acc:>7.2f}% | {prec:>8.2f}% | {rec:>7.2f}% | {f1:>7.2f}%")
                    st.session_state.models[name] = model
                    st.session_state.log_tab2 += f"      ✓ Done.\n"

                st.session_state.log_tab2 += "\n[5/5] Packaging and Saving Models...\n"
                save_filename = f"phishguard_models_{len(df)}_records.pkl"
                save_path = os.path.join(JOBLIBS_DIR, save_filename)
                
                bundle = {
                    'vectorizer': st.session_state.vectorizer,
                    'models': st.session_state.models
                }
                joblib.dump(bundle, save_path)
                st.session_state.log_tab2 += f"   💾 Successfully saved to '{save_path}'\n"

                table_header = f"\n{'Algorithm':<15} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8}"
                st.session_state.log_tab2 += "-" * 65 + table_header + "\n" + "-" * 65 + "\n"
                for row in results:
                    st.session_state.log_tab2 += row + "\n"
                st.session_state.log_tab2 += "-" * 65 + "\n\n✅ TRAINING COMPLETE. Models are ready in Tab 4."
                
                st.session_state.flow_step = 4
                log_placeholder_2.text_area("Training Logs", st.session_state.log_tab2, height=350)
                time.sleep(1)
                st.rerun()

            except Exception as e:
                st.error(f"[CRITICAL ERROR] Pipeline failed: {e}")

    elif st.session_state.log_tab2:
        log_placeholder_2.text_area("Training Logs", st.session_state.log_tab2, height=350)

# -----------------------------------------------------------------------
# TAB 3: DATA SAMPLER
# -----------------------------------------------------------------------
with tab3:
    st.header("🧪 Data Sampler & Synthetic Generator")
    
    sampler_csv = st.selectbox("Select Dataset to Analyze:", ["No CSV found"] if not csv_files else csv_files, key="sampler_box")
    
    colX, colY, colZ = st.columns(3)
    extract_phish = colX.button("🔴 Extract Real Phishing", use_container_width=True)
    extract_safe = colY.button("🟢 Extract Real Safe (Ham)", use_container_width=True)
    extract_synth = colZ.button("🤖 Generate Synthetic Phishing", use_container_width=True)

    log_placeholder_3 = st.empty()

    def load_sampler_df(filename):
        if filename == "No CSV found": return None
        try:
            # Append the directory path for the sampler
            file_path = os.path.join(DATASETS_DIR, filename)
            df = pd.read_csv(file_path)
            if not {'text_content', 'label'}.issubset(df.columns): return None
            return df
        except: return None

    if extract_phish or extract_safe:
        df = load_sampler_df(sampler_csv)
        if df is not None:
            label_type = 1 if extract_phish else 0
            filtered_df = df[df['label'] == label_type]
            if not filtered_df.empty:
                count = min(3, len(filtered_df))
                samples = filtered_df.sample(n=count)['text_content'].tolist()
                tag = "🔴 PHISHING (Malicious)" if label_type == 1 else "🟢 HAM (Safe/Legitimate)"
                
                out = f"--- Extracting {count} Real {tag} Samples ---\n\n"
                for i, text in enumerate(samples, 1):
                    out += f"[Sample {i}]\n > {text}\n\n"
                log_placeholder_3.info(out)
            else:
                st.warning("No matching records found.")

    if extract_synth:
        df = load_sampler_df(sampler_csv)
        if df is not None:
            phishing_texts = df[df['label'] == 1]['text_content'].dropna().tolist()
            if phishing_texts:
                with st.spinner("AI is generating synthetic text..."):
                    markov_dict = {}
                    for text in phishing_texts:
                        words = str(text).split()
                        for i in range(len(words) - 1):
                            current_word = words[i]
                            next_word = words[i + 1]
                            if current_word not in markov_dict:
                                markov_dict[current_word] = []
                            markov_dict[current_word].append(next_word)

                    out = "--- Generating Synthetic Phishing via AI Markov Chain ---\n\n"
                    for i in range(1, 4):
                        if not markov_dict: break
                        capitalized_words = [w for w in markov_dict.keys() if w[0].isupper()]
                        current_word = random.choice(capitalized_words) if capitalized_words else random.choice(list(markov_dict.keys()))
                        sentence = [current_word]
                        
                        for _ in range(19):
                            if current_word in markov_dict and markov_dict[current_word]:
                                next_word = random.choice(markov_dict[current_word])
                                sentence.append(next_word)
                                current_word = next_word
                            else: break
                        out += f"[Synthetic Phish {i}]\n > {' '.join(sentence)}...\n\n"
                    log_placeholder_3.success(out)

# -----------------------------------------------------------------------
# TAB 4: LIVE DETECTION PROTOTYPE
# -----------------------------------------------------------------------
with tab4:
    st.header("🛡️ PhishGuard Live Scanner")

    if not st.session_state.models:
        st.warning("⚠️ Train or Load AI Models in Tab 2 before using the Live Scanner.")
    else:
        active_model_name = st.selectbox("Active AI Engine:", list(st.session_state.models.keys()))
        
        st.markdown("### Option 1: Manual Text Input")
        raw_text_input = st.text_area("Paste suspicious Email or SMS text:", height=100)
        analyze_text_btn = st.button("🔍 Analyze Text")
        
        st.divider()
        
        st.markdown("### Option 2: Live URL Scanner")
        url_input = st.text_input("Live Crawl a Suspicious Website or Forum URL:", placeholder="https://example.com")
        analyze_url_btn = st.button("🌐 Crawl & Analyze URL", type="primary")

        # --- Prediction Logic ---
        def execute_prediction(raw_text):
            if not raw_text.strip():
                st.warning("Please enter some text or a valid URL.")
                return

            model = st.session_state.models[active_model_name]
            text_lower = raw_text.lower().translate(str.maketrans('', '', string.punctuation))
            stops = set(stopwords.words('english'))
            cleaned = " ".join([w for w in text_lower.split() if w not in stops])
            
            vectorized_text = st.session_state.vectorizer.transform([cleaned])
            prediction = model.predict(vectorized_text)[0]

            if prediction == 1:
                st.error("### 🔴 WARNING: PHISHING THREAT DETECTED!")
            else:
                st.success("### 🟢 SAFE: No malicious intent detected.")

        if analyze_text_btn:
            execute_prediction(raw_text_input)

        if analyze_url_btn:
            if not url_input.startswith("http"):
                st.warning("Invalid URL. Must start with http:// or https://")
            else:
                with st.spinner("⚙️ Scraping Live URL..."):
                    try:
                        options = Options()
                        options.add_argument("--headless")
                        options.add_argument("--disable-gpu")
                        options.add_argument("--log-level=3")
                        
                        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
                        driver.get(url_input)
                        time.sleep(2)
                        page_text = driver.find_element(By.TAG_NAME, "body").text
                        driver.quit()

                        if len(page_text) > 10000:
                            page_text = page_text[:10000]

                        execute_prediction(page_text)
                    except Exception as e:
                        st.error("⚠️ Scraping Failed. Is the URL reachable?")