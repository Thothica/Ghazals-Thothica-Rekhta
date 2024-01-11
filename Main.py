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

query = st.text_input(label = 'Please enter your query - ', value = 'lover oh lover, where are you?')
top_k = st.number_input(label = 'Top k - ', min_value = 3, max_value = 25, value = 5)

retriever = create_retriever()

if query and top_k:
    col1, col2 = st.columns([3, 2])
    with col1:
        response = []
        for i in retriever.retrieve(query):
            response.append({
                    'Content_Ur' : i.node.metadata['Content_Ur'],
                    'Name_Ur' : i.node.metadata['Name_Ur'],
                    'Content_Hi' : i.node.metadata['Content_Hi'],
                    'Name_Hi' : i.node.metadata['Name_Hi'],
                    'Content_En' : i.node.metadata['Content_En'],
                    'Name_En' : i.node.metadata['Name_En'],
                    'Text' : i.get_text(),
                    'Score' : i.get_score(),
                })
        st.json(response)
    
    with col2:
        summary = st.empty()
        top3 = []
        top3_couplet = []
        top3_name = []
        for i in response:
             top3.append(i["Text"])
             top3_couplet.append(i["Content_Ur"])
             top3_name.append(i["Name_En"])
        temp_summary = []
        for resp in client.chat.completions.create(model = "gpt-4-1106-preview",
            messages = [
                    {"role": "system", "content": "Act as a Shayari Interpretation Summarizer GPT. The GPT's primary role is to provide succinct summaries of interpretations of Urdu couplets in relation to the user's queries, focusing on the context of the question and providing insights about the poet and the poem. It will deliver detailed responses, aiming for around 500 words, to enrich the user's understanding. Integrated into Rekhta's website, the GPT will avoid lengthy literary critiques or personal opinions. It will make an informed guess when the query is ambiguous or the couplet is particularly complex, ensuring accurate and informative responses while maintaining a respectful tone, reflective of Urdu poetry's literary and cultural richness. It won't ask user for clarification so as to give a seamless experience."},
                    {"role": "user", "content": f"""Summarize the following interpretation of couplets in context of the query “{query}”:

{top3_name[0]}
{top3_couplet[0]}
Interpretation:
{top3[0]}

{top3_name[1]}
{top3_couplet[1]}
Interpretation:
{top3[1]}

{top3_name[2]}
{top3_couplet[2]}
Interpretation:
{top3[2]}"""},
                ],
            stream = True):
                if resp.choices[0].finish_reason == "stop":
                    break
                temp_summary.append(resp.choices[0].delta.content)
                result = "".join(temp_summary).strip()    
                summary.markdown(f'{result}')
