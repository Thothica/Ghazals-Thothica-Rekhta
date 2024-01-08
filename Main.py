import streamlit as st
import os
from llama_index import load_index_from_storage, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from openai import OpenAI

st.set_page_config(layout = "wide")

client = OpenAI(api_key = st.secrets['OPENAI_API_KEY'])
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

@st.cache_resource
def create_retriever():
    index = load_index_from_storage(storage_context = StorageContext.from_defaults(
                docstore = SimpleDocumentStore.from_persist_dir(persist_dir = "storage"),
                vector_store = FaissVectorStore.from_persist_dir(persist_dir = "storage"),
                index_store = SimpleIndexStore.from_persist_dir(persist_dir = "storage"),
            ))
    return index.as_retriever(retriever_mode = 'embedding', similarity_top_k = int(top_k))

st.title('Thothica Rekhta Ghazal Search')

query = st.text_input(label = 'Please enter your query - ', value = 'Shayari about gifting')
top_k = st.number_input(label = 'Top k - ', min_value = 2, max_value = 25, value = 5)

retriever = create_retriever()

if query and top_k:
    col1, col2 = st.columns([3, 2])
    with col1:
        response = []
        for i in retriever.retrieve(query):
            response.append({
                    'Text' : i.get_text(),
                    'Score' : i.get_score(),
                    'Name_En' : i.node.metadata['Name_En'],
                    'Name_Ur' : i.node.metadata['Name_Ur'],
                    'Name_Hi' : i.node.metadata['Name_Hi'],
                    'Content_En' : i.node.metadata['Content_En'],
                    'Content_Ur' : i.node.metadata['Content_Ur'],
                    'Content_Hi' : i.node.metadata['Content_Hi'],
                })
        st.json(response)
    
    with col2:
        summary = st.empty()
        top3 = ""
        for i in response:
             top3 += i["Text"] + "\n"
        temp_summary = []
        for resp in client.chat.completions.create(model = "gpt-4",
            messages = [
                    {"role": "system", "content": "You are an eloquent Ghazal artist."},
                    {"role": "user", "content": f"You will be given three top ghazals based on a prompt summarise them to make an answer \n\nGhazals-{top3}\n\nPrompt-{query}"},
                ],
            stream = True):
                if resp.choices[0].finish_reason == "stop":
                    break
                temp_summary.append(resp.choices[0].delta.content)
                result = "".join(temp_summary).strip()    
                summary.markdown(f'{result}')
