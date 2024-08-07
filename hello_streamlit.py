import streamlit as st
from langchain import hub
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_response(uploaded_file, base_url, api_key, model, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.create_documents(documents)
        llm = ChatOpenAI(base_url=base_url, model=model, api_key=api_key)
        # Select embeddings
        embeddings = OpenAIEmbeddings(base_url=base_url, model="text-embedding-3-small", api_key=api_key)
        # Create a vectorstore from documents
        database = Chroma.from_documents(texts, embeddings)
        # Create retriever interface
        retriever = database.as_retriever()
        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
)
        # Create QA chain
        response = rag_chain.invoke(query_text)
        return response

# File upload
uploaded_file = st.file_uploader('Upload an article', type='txt')
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = None
with st.form('myform', clear_on_submit=False, border=False):
    api_key = st.text_input('AGICTO API Key', type='password', disabled=not (uploaded_file and query_text))
    base_url = "https://api.fe8.cn/v1"
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted and api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, base_url, api_key, "deepseek-chat", query_text)
            result = response
if result:
    st.info(result)