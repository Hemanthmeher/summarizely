"""
Website Summarizer Application
"""

import traceback
from io import BytesIO

import validators
import streamlit as st

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader


# Page Configuration
st.set_page_config(
    page_title="Website Summarizer",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Header Section
st.title("Website Summarizer")
st.markdown(
    """
Extract content from any web page and generate structured summaries using Groq and LangChain.

**Instructions:**
1. Enter your Groq API Key in the sidebar.
2. Provide the target website URL below.
3. Select summary length and chain type, then click Summarize.
"""
)

# Sidebar Configuration
with st.sidebar:
    st.header("Settings")
    
    groq_api_key = st.text_input("Groq API Key", value="", type="password", placeholder="gsk_...")
    summary_words = st.slider("Target summary length (words)", min_value=50, max_value=1000, value=300, step=50)
    show_extracted = st.checkbox("Show raw text preview", value=False)
    chain_type = st.selectbox("Summarization chain type", options=["stuff", "map_reduce", "refine"], index=0)
    
    st.markdown("---")
    st.caption("Supports public web articles and text pages. Ensure target sites permit content scraping.")
    
    dev_mode = st.checkbox("Enable debug mode", value=False)

st.session_state["dev_mode"] = dev_mode

# Main Input Area
st.markdown("### Target URL")
generic_url = st.text_input("Website URL", placeholder="https://example.com/article", label_visibility="collapsed")

col1, col2 = st.columns([3, 1])
with col1:
    summarize_btn = st.button("Summarize website", use_container_width=True)
with col2:
    clear_btn = st.button("Clear", use_container_width=True)

if clear_btn:
    st.rerun()

# Helper Functions
@st.cache_data(show_spinner=False)
def load_url_content(url: str):
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
    template = (
        f"Provide a concise, professional summary of the following web page content in approximately {words} words.\n\n"
        "Content:\n{text}\n\n"
        "Highlight key bullet points and main conclusions with a neutral, objective tone."
    )
    return PromptTemplate(template=template, input_variables=["text"])


# Main Execution Logic
if summarize_btn:
    if not groq_api_key.strip():
        st.sidebar.error("Please enter your Groq API key.")
    elif not generic_url.strip():
        st.error("Please enter a website URL.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL (including http:// or https://).")
    else:
        try:
            with st.spinner("Fetching web page content..."):
                docs = load_url_content(generic_url)

            if not docs:
                st.error("No extractable content found at the provided URL.")
            else:
                if show_extracted:
                    st.subheader("Extracted Text Preview")
                    preview = "\n\n".join([d.page_content[:5000] for d in docs])
                    st.text_area("Raw Content", value=preview, height=250)

                llm = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=groq_api_key)
                prompt = make_prompt(summary_words)

                with st.spinner("Generating summary..."):
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
                        st.info(f"Debug: Loaded docs: {len(docs)} | First doc length: {first_len} chars | Chain: {chain_type}")

                st.success("Summary generated successfully.")
                st.subheader("Executive Summary")
                st.write(output_summary)

                b = BytesIO()
                b.write(output_summary.encode("utf-8"))
                b.seek(0)
                st.download_button(
                    label="Download summary (.txt)",
                    data=b,
                    file_name="summary.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Processing error: {e}")
            if st.session_state.get("dev_mode", False):
                st.exception(traceback.format_exc())

# Footer
st.markdown("---")
st.caption("Developed with Streamlit, LangChain, and Groq.")