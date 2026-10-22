"""
🚀 AI Website Summarizer — LangChain + Groq + Streamlit
"""

import traceback
from io import BytesIO

import validators
import streamlit as st

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader

st.set_page_config(
    page_title="AI Website Summarizer Pro",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title(" AI Website Summarizer Pro")
st.markdown(
    """
An advanced, lightweight AI application that extracts content from any web page and generates concise, structured summaries using **Groq (LLaMA)** and **LangChain**.

**How to use:**
1. Enter your **Groq API Key** in the sidebar settings.
2. Paste any readable website link (URL) below.
3. Choose your desired summary length and summarization method, then click **Summarize website**.
"""
)

with st.sidebar:
    st.header("⚙️ Application Settings")

    groq_api_key = st.text_input("Groq API Key", value="", type="password", placeholder="gsk_...")

    summary_words = st.slider("Approx. summary length (words)", min_value=50, max_value=1000, value=300, step=50)
    
    show_extracted = st.checkbox("Show extracted page text preview", value=False)
    chain_type = st.selectbox("Summarization chain type", options=["stuff", "map_reduce", "refine"], index=0)
    
    st.markdown("---")
    st.caption("💡 Supports general websites and articles (No YouTube). Please ensure target websites allow scraping.")

    dev_mode = st.checkbox("🛠️ Dev mode (Show Debug info)", value=False)

st.session_state["dev_mode"] = dev_mode
st.markdown("### 🌐 Target URL")
generic_url = st.text_input("Website URL", placeholder="https://example.com/article-to-summarize", label_visibility="collapsed")

col1, col2 = st.columns([3, 1])
with col1:
    summarize_btn = st.button("🚀 Summarize website", use_container_width=True)
with col2:
    clear_btn = st.button("🔄 Clear", use_container_width=True)

if clear_btn:
    st.rerun()

@st.cache_data(show_spinner=False)
def load_url_content(url: str):
    """
    Load website content using UnstructuredURLLoader and return list of LangChain docs.
    """
    loader = UnstructuredURLLoader(
        urls=[url],
        ssl_verify=False,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        },
    )
    docs = loader.load()
    return docs


def make_prompt(words: int) -> PromptTemplate:
    """
    Build a custom PromptTemplate for summary length guidance and structure.
    """
    template = (
        f"Provide a concise, professional, and well-structured summary of the following web page content in about {words} words.\n\n"
        "Content:\n{text}\n\n"
        "Highlight key bullet points, main insights, and clear conclusions. Keep the tone informative and neutral."
    )
    return PromptTemplate(template=template, input_variables=["text"])


if summarize_btn:
    # Basic input validation checks
    if not groq_api_key.strip():
        st.sidebar.error("⚠️ Please enter your Groq API key in the sidebar.")
    elif not generic_url.strip():
        st.error("⚠️ Please enter a website URL to summarize.")
    elif not validators.url(generic_url):
        st.error("⚠️ Please enter a valid URL (ensure it includes http:// or https://).")
    else:
        try:
            with st.spinner("📥 Fetching web page content..."):
                docs = load_url_content(generic_url)

            if not docs:
                st.error("❌ No readable textual content could be extracted from the provided URL.")
            else:
                # Optionally display extracted text preview
                if show_extracted:
                    st.subheader("📄 Extracted Text Preview")
                    preview = "\n\n".join([d.page_content[:5000] for d in docs])
                    st.text_area("Raw Text Content", value=preview, height=250)

                # Initialize Groq LLM with a reliable, fast model
                llm = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=groq_api_key)

                # Build prompt template
                prompt = make_prompt(summary_words)

                with st.spinner("🤖 Summarizing content with Groq LLM..."):
                    # Fix: Handle different chain types correctly without causing Pydantic extra field errors
                    if chain_type == "stuff":
                        chain = load_summarize_chain(llm, chain_type=chain_type, prompt=prompt)
                        output_summary = chain.run(input_documents=docs)
                    else:
                        chain = load_summarize_chain(llm, chain_type=chain_type)
                        output_summary = chain.run(input_documents=docs)

                    if st.session_state.get("dev_mode", False):
                        try:
                            first_len = len(docs[0].page_content) if docs else 0
                        except Exception:
                            first_len = -1
                        st.info(f"🛠️ [Debug Info] Docs loaded: {len(docs)} | First doc character length: {first_len} | Chain Type: {chain_type}")

                st.success("✨ Summary generated successfully!")
                st.subheader("📌 Executive Summary")
                st.write(output_summary)

                b = BytesIO()
                b.write(output_summary.encode("utf-8"))
                b.seek(0)
                st.download_button(
                    label="📥 Download summary as .txt",
                    data=b,
                    file_name="website_summary.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"❌ An error occurred during processing: {e}")
            if st.session_state.get("dev_mode", False):
                st.exception(traceback.format_exc())

# Footer Credits
st.markdown("---")
st.caption("🚀 Developed with Streamlit, LangChain, and Groq | AI Engineering Portfolio Project")