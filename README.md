# 🌐 Chat with Website

A lightweight, high-speed AI application to chat with any website URL using **LangChain**, **Groq**, and **Streamlit**.

## 🚀 Overview

Ingests any website URL on-the-fly and enables conversational Q&A with full chat memory. Built with local in-memory BM25 retrieval (zero external API calls or embedding quotas during ingestion) paired with ultra-fast Groq open-source LLM inference.

## ✨ Features

- **Any Website URL**: Scrapes and indexes webpage content dynamically.
- **Zero-Quota Ingestion**: In-memory BM25 retrieval eliminates rate limits and embedding costs.
- **Groq LPU Inference**: Powered by active open-source models (`openai/gpt-oss-20b`, `qwen/qwen3.8-27b`).
- **Conversational Memory**: History-aware query reformulator that understands follow-up context.
- **Streamlit UI**: Clean, responsive chat interface.

## 🛠️ Quickstart

### 1. Clone & Install
```bash
git clone https://github.com/vishalengineerai/Chat_With_Website.git
cd Chat_With_Website
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Key
```bash
cp .env.example .env
```
Add your free Groq API key from [console.groq.com/keys](https://console.groq.com/keys):
```env
GROQ_API_KEY=gsk_your_key_here
```

### 3. Run
```bash
streamlit run app.py
```
App runs at `http://localhost:8501`.
