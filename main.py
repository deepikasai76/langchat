import os
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from InstructorEmbedding import INSTRUCTOR
from langchain.llms import HuggingFaceHub
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from tqdm.autonotebook import trange
from tqdm import tqdm
import pickle
import transformers
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
import hnswlib
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma
import os
from langchain.document_loaders import DirectoryLoader
from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import os
import torch
from sqlite3 import Connection
import pdf2image
import PIL
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTContainer, LTImage, LTItem, LTTextBox
from pdfminer.utils import open_filename
import sys
import json
import os

os.environ['OPENAI_API_KEY'] = "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"
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"



# Load conversation history from a JSON file
def load_conversation_history():
    try:
        with open("conversation_history.json", "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

# Save conversation history to a JSON file
def save_conversation_history(conversation):
    with open("conversation_history.json", "w") as file:
        json.dump(conversation, file)

def main():
    # Set Streamlit page configuration
    st.set_page_config(page_title="Langchain PDF Chatbot 🤖", layout='centered')
    # Initialize session states
    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "past" not in st.session_state:
        st.session_state["past"] = []
    if "input" not in st.session_state:
        st.session_state["input"] = ""
    if "stored_session" not in st.session_state:
        st.session_state["stored_session"] = []

    st.header("🦜️🔗LangChain Chatbot🤖 Enhanced by OpenAI for Merged Approved PDF Documents")
    query = None

    _template = """Given the following conversation and a follow up question,
    rephrase the follow up question to be a standalone question.

    Chat History:

    {chat_history}

    Follow Up Input: {question}

    Standalone question:"""

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    prompt_template = """You are an AI assistant whose name is Energy Buddy and you will answer questions from the relevant vectorstore embeddings provided in context . 
    Provide a conversational answer from the context and If you are asked about anything else than Energy , 
    just say that you are not allowed to talk about it, don't try to make up an answer.
    {context}
    Question: {question}
    Helpful Answer: """

    QA_PROMPT = PromptTemplate( template = prompt_template, input_variables=["context", "question"])


    # Clear the conversation history file at the beginning of each run
    if os.path.exists("conversation_history.json"):
        os.remove("conversation_history.json")

    # Enable to save to disk & reuse the model (for repeated queries on the same data)
    PERSIST = True
    pdf_folder_path = f'C:/Users/home/Desktop/lang_chat/pdf/'
    st.write(pdf_folder_path)
    os.listdir(pdf_folder_path)

    if PERSIST and os.path.exists("persist"):
        st.write("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
    #loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
    # loader = DirectoryLoader("pdf_folder_path/")
    # Iterate through the files in the pdf_folder_path
        for fn in tqdm(os.listdir(pdf_folder_path)):
            full_path = os.path.join(pdf_folder_path, fn)
            st.write(full_path)
        loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in tqdm(os.listdir(pdf_folder_path))]

        if PERSIST:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders(loaders)
        else:
            index = VectorstoreIndexCreator().from_loaders(loaders)

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(qa_prompt=QA_PROMPT,verbose=True,streaming=True, search_kwargs={"k": 1}),
    )


    chat_history = load_conversation_history()

    while True:

        if not query:
            query = st.text_input("User: ",
                                  value=" ",
                                  placeholder="I am your AI assistant! Ask me anything about your Document...",
                                label_visibility='hidden')
            

        if query and query.lower() in ['quit', 'q', 'exit']:
            break

        if query:
            result = chain({"question": query, "chat_history": chat_history})
            chat_history.append({"user": query, "bot": result['answer']})
            st.write("Bot:", result['answer'])
            st.session_state.past.append(query)
            st.session_state.generated.append(result["answer"])

            if st.session_state["generated"]:
                # Display conversation history
                st.markdown("<h2 style='font-size: 30px; font-weight: bold;'>Conversation History:</h2>",
                            unsafe_allow_html=True)

                for i in range(len(st.session_state["generated"]) - 1, -1, -1):

                    st.write("User:", st.session_state["past"][i], is_user=True,avatar_style="thumbs",seed='Aneka', key = str(i) +"_user")
                    st.write("Bot🤖:", st.session_state["generated"][i], avatar_style="fun-emoji",key=str(i))
                    st.write("" * 20)
            else:
                st.session_state.clear()


        # # Append the new question and answer to the conversation history
        # chat_history.append({"user": query, "bot": result['answer']})

        # query = None

        # # Display previous questions and answers in the UI
        # st.write("Conversation History:")
        # for entry in chat_history:
        #     st.write("User:", entry["user"])
        #     st.write("Bot🤖:", entry["bot"])
        #     st.write("-" * 20)
        #

        # Save the updated conversation history
        save_conversation_history(chat_history)



if __name__ == '__main__':
    main()
