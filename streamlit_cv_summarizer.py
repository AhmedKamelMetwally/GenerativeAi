import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.document_loaders import PyPDFLoader
from pydantic import BaseModel, ValidationError
import tempfile
import json
import os
from typing import Optional

class Resume(BaseModel):
    first_name: str
    last_name: str
    email: Optional[str] = None
    university: Optional[str] = None
    degree: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None


prompt_template = ChatPromptTemplate.from_template("""
You are an AI that extracts structured resume data from unstructured text.

Given the resume content below, extract the following fields in JSON format:
- first_name
- last_name
- email
- university
- degree
- company
- job_title

Resume:
{text}

Return only valid JSON.
""")


llm = ChatOllama(model="gemma")


st.set_page_config(page_title="Resume Parser", layout="centered")
st.title("📄 Resume Parser AI")
st.markdown("Upload a resume PDF and extract structured information.")


uploaded_file = st.file_uploader("Upload PDF Resume", type=["pdf"])

if uploaded_file:
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    
    loader = PyPDFLoader(tmp_path)
    docs = loader.load_and_split()
    text = "\n".join([doc.page_content for doc in docs])

    
    os.remove(tmp_path)

    with st.spinner("Extracting information..."):
        
        chain = prompt_template | llm
        response = chain.invoke({"text": text})

        
        try:
            result_json = json.loads(response.content)
            resume = Resume(**result_json)
            st.success("✅ Resume extracted successfully!")

            # Show parsed information
            st.subheader("📋 Parsed Information")
            st.json(resume.dict(), expanded=True)

        except (json.JSONDecodeError, ValidationError) as e:
            st.error("")
            st.text(str(e))
            st.text(response.content)