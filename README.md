<div align="center">
  <img src="logo/logo_clear.png" width="120" alt="PhishGuard AI Logo">
  
  # 🛡️ PhishGuard AI Scanner
  **Advanced Natural Language Processing for Threat Detection**

  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
  [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
</div>

---

## 📖 Overview
**PhishGuard AI** is a comprehensive, end-to-end machine learning pipeline and interactive web dashboard designed to identify, analyze, and mitigate phishing threats and malicious communications. 

Built with an emphasis on transparency and threat intelligence, the system handles everything from automated data scraping and Natural Language Processing (NLP) to model training and live URL scanning.

## ✨ Core Features
The dashboard is divided into four primary modules:

* **🌐 1. Data Gathering:** Automated web scraping engine using headless Selenium. Ingests raw `.csv`/`.tsv` files or live URLs, executes smart column detection to isolate labels/text, and standardizes data for NLP pipelines.
* **🧠 2. AI Training:** End-to-end model compilation. Processes text via NLTK (stopword removal) and TF-IDF vectorization. Trains and evaluates three separate classification engines: **Naïve Bayes**, **Support Vector Machines (Linear)**, and **Random Forest**.
* **🧪 3. Sampler Sandbox:** Threat intelligence module featuring a Markov Chain text generator to synthesize artificial phishing attempts for system testing and analysis.
* **🛡️ 4. Live Detection:** Real-time threat scanner. Evaluates manual text inputs or actively scrapes live URLs to classify malicious intent using the pre-trained NLP models.

## 🏗️ Technical Architecture
* **Frontend UI:** Streamlit
* **Machine Learning:** Scikit-Learn, Joblib
* **Natural Language Processing:** NLTK (Stopwords), TF-IDF
* **Data Extraction/Scraping:** Selenium WebDriver, Pandas

---

## 💻 Local Installation

To run this project on your local machine, ensure you have Git and Python installed.

**1. Clone the repository & setup LFS**
*(Note: This project uses Git Large File Storage for compiled `.pkl` model weights).*
```bash
git clone [https://github.com/TheRealSRZ/PhishGuard.git](https://github.com/TheRealSRZ/PhishGuard.git)
cd PhishGuard
git lfs install
git lfs pull
