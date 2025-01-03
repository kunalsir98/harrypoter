import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load Groq API Key
groq_api_key = os.getenv('GROQ_API_KEY')

# Harry Potter-themed content (expanded context)
harry_potter_content = """
The Harry Potter series is a collection of seven fantasy novels written by J.K. Rowling. The main plot revolves around a young wizard named Harry Potter and his adventures at Hogwarts School of Witchcraft and Wizardry. 

Key Characters:
- **Harry Potter**: The protagonist, known as "The Boy Who Lived". He survived an attack by the dark wizard Voldemort as a baby and is famous for defeating him as a child.
- **Hermione Granger**: Harry's intelligent and resourceful best friend. She is known for her dedication to learning and her powerful spell-casting abilities.
- **Ron Weasley**: Harry's loyal friend who comes from a large wizarding family. He is known for his bravery and loyalty to his friends.
- **Lord Voldemort**: The dark wizard who seeks to conquer the wizarding world and kill Harry Potter. His real name is Tom Riddle.
- **Albus Dumbledore**: The wise and kind Headmaster of Hogwarts. He is a mentor to Harry and plays a key role in the fight against Voldemort.
- **Severus Snape**: The complex and enigmatic Potions Master at Hogwarts. He is revealed to have played a key role in both sides of the war.
- **Draco Malfoy**: A student at Hogwarts and Harry's rival. He is part of the Malfoy family, who are loyal to Voldemort.

Key Spells:
- **Expelliarmus**: A disarming spell used to knock away an opponent's weapon or force them to release their hold on an object.
- **Lumos**: A spell used to light the tip of a wand, functioning like a flashlight.
- **Avada Kedavra**: The Killing Curse, one of the Unforgivable Curses, used to instantly kill its target.
- **Expecto Patronum**: A charm that summons a Patronus, a magical creature used to ward off Dementors.
- **Wingardium Leviosa**: A levitation charm that is used to make objects float.

Hogwarts Houses:
- **Gryffindor**: Known for bravery and courage, its emblem is a lion, and its colors are red and gold. Famous members include Harry Potter, Hermione Granger, and Ron Weasley.
- **Slytherin**: Known for ambition and cunning, its emblem is a serpent, and its colors are green and silver. Famous members include Draco Malfoy and Severus Snape.
- **Hufflepuff**: Known for loyalty and hard work, its emblem is a badger, and its colors are yellow and black. Famous members include Cedric Diggory and Newt Scamander.
- **Ravenclaw**: Known for wisdom and wit, its emblem is an eagle, and its colors are blue and silver. Famous members include Luna Lovegood and Cho Chang.

Key Events:
- **The Battle of Hogwarts**: The final showdown between the forces of good (led by Harry and his allies) and Voldemort's Death Eaters.
- **The Triwizard Tournament**: A magical competition held at Hogwarts, in which Harry participates despite not being a chosen champion.
- **The Philosopher's Stone**: The first book in the series, where Harry discovers his magical heritage and stops Voldemort from obtaining immortality.

The Harry Potter series also touches on themes of friendship, sacrifice, and the importance of choices over destiny. The books explore how individuals can overcome prejudice, fight for justice, and grow as people.

"""

from langchain.schema import Document

# Initialize session state
if 'vector' not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings()
    st.session_state.loader = WebBaseLoader('https://en.wikipedia.org/wiki/Harry_Potter')
    
    # Wrap the harry_potter_content into Document objects
    harry_potter_documents = [Document(page_content=harry_potter_content)]
    
    # Now you can split documents
    st.session_state.docs = harry_potter_documents
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Title and header
st.markdown(
    """
    <style>
        body { background-color: #0e1117; color: white; }
        .title { font-size: 50px; font-weight: bold; text-align: center; color: #61dafb; }
        .subheader { font-size: 20px; text-align: center; margin-bottom: 30px; color: #adb5bd; }
        .response-box { padding: 20px; background-color: #1a1f27; border-radius: 10px; color: #ffffff; margin-top: 20px; }
        .document-box { margin: 10px 0; padding: 10px; background-color: #2b2f38; border-radius: 5px; color: #e9ecef; }
        .expander { background-color: #1c1e26; border-radius: 5px; padding: 10px; margin-top: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">Harry Potter ChatBot</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Ask me anything about the Harry Potter universe!</div>', unsafe_allow_html=True)

# Initialize LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name='Llama3-8b-8192'
)

# Define prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question on provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    Question: {input}
    """
)

document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Prompt input
user_input = st.text_input(
    "Ask your question:",
    placeholder="Who is the main character in Harry Potter?",
    help="Enter any question about the Harry Potter universe."
)

# Response section
if user_input:
    start_time = time.process_time()
    response = retrieval_chain.invoke({"input": user_input})
    response_time = time.process_time() - start_time

    # Display the AI's response
    st.markdown(f'<div class="response-box">🧠 **Response:** {response["answer"]}</div>', unsafe_allow_html=True)

    # Show additional document similarity search
    with st.expander("Document similarity search", expanded=False):
        for doc in response['context']:
            st.markdown(f'<div class="document-box">{doc.page_content}</div>', unsafe_allow_html=True)
            st.write(" ")

    # Display response time
    st.markdown(f"**Response Time:** {response_time:.2f} seconds")

# Footer
st.markdown(
    """
    <style>
        footer { font-size: 12px; text-align: center; margin-top: 50px; color: #adb5bd; }
    </style>
    <footer>Powered by LangChain & Streamlit | Designed with 💻 by [kunal]</footer>
    """,
    unsafe_allow_html=True
)
