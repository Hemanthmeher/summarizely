# 🤖 Summarizely — AI Website Summarizer Pro

An advanced, lightweight AI-powered web application built with **Streamlit**, **LangChain**, and **Groq (LLaMA)** that extracts clean text content from any public web page and generates concise, structured executive summaries.

---

## 🚀 About The Project

Summarizely is designed to help users quickly distill long articles, documentation, or web pages into key insights without reading through blocks of text. By combining **LangChain's** robust document loaders and summarization chains with ultra-fast **Groq LLM inference**, this tool delivers accurate, customizable summaries in seconds.

---

## ✨ Key Features

- **URL Content Extraction**: Leverages LangChain's `UnstructuredURLLoader` to securely scrape clean text from target web pages.
- **Customizable Chain Types**: Supports multiple LangChain summarization strategies (`stuff`, `map_reduce`, `refine`) for different content lengths.
- **Adjustable Word Count**: Seamlessly control the length of your summary using an interactive sidebar slider (50 to 1000 words).
- **Text Preview Option**: Inspect the raw extracted text before generating the final executive summary.
- **Export Feature**: Download generated summaries instantly as a `.txt` file with a single click.
- **Developer Debug Mode**: Built-in debug panels and error handling for easy troubleshooting.

---

## 🛠️ Tech Stack

- **Frontend / UI**: [Streamlit](https://streamlit.io/)
- **Orchestration**: [LangChain](https://www.langchain.com/)
- **LLM Engine**: [Groq API](https://groq.com/) (LLaMA 3)
- **Utilities**: Python, Validators, BytesIO

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Hemanthmeher/summarizely.git](https://github.com/Hemanthmeher/summarizely.git)
   cd summarizely
Create and activate a virtual environment:

Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
Install dependencies:

Bash
pip install -r requirements.txt
Run the Streamlit application:

Bash
streamlit run app.py
💡 How to Use
Enter your Groq API Key in the sidebar settings.

Paste any readable website URL into the target input box.

Configure your summary length and summarization method.

Click 🚀 Summarize website and download your summary if needed!
