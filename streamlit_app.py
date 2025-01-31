#!/usr/bin/env python
# coding: utf-8

# In[1]:


from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
import streamlit as st


# In[2]:


# In[3]:


# 1. Define all the LLMs to be used
embed_llm = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

query_llm = Anthropic(
                model="claude-3-5-haiku-20241022",
                temperature=0.7,
                system_prompt="""You are Niko Canner, an entrepreneur,investor, philosopher, thought leader, and excellent writer.
                Return your answers in language that is accessible, concise, precise, but insightful. 
                Write the response in first person in the voice of Niko. Keep the tone similar to the original text
                that you are summarizing. Each response you give is short, no more than 300 words max.
                For every response, list the titles of the sources you drew the response from at the end. If you
                don't find any sources, write "None" in the sources list.
                """
)


# In[4]:


Settings.llm = query_llm
Settings.embed_model = embed_llm
Settings.chunk_size = 512 #limit of our chosen embedding model
Settings.chunk_overlap = 100


# In[5]:


# 2. Define the RAG vector database
def load_posts():
    reader = SimpleDirectoryReader(input_dir="./niko_posts/")
    blog_posts = reader.load_data()    
    index = VectorStoreIndex.from_documents(blog_posts,embed_model=embed_llm)
    return index


# In[6]:


index = load_posts()


# In[7]:


# Streamlit setup

st.set_page_config(page_title="Chat with Niko's blog posts", layout="centered", initial_sidebar_state="auto", menu_items=None)
#openai.api_key = st.secrets.openai_key
st.title("Chat with Niko 💬")
st.info("Demo of a RAG chatbot powered by Claude")
if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask Niko a question!",
        }
    ]


# In[8]:


memory = ChatMemoryBuffer.from_defaults(token_limit=15000)


# In[9]:


if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        streaming=True
    )


# In[12]:


if prompt := st.chat_input(
    "Ask a question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})


# In[13]:


for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])


# In[14]:


# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)


# In[ ]:





