from openai import OpenAI
import streamlit as st
from model import create_model
from database import create_database

vector_db = create_database()
model = create_model(vector_db)

st.title("💬 Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Olá, sou o ajudante para o Enem 2024. Em que posso lhe ajuda?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    msg = model.invoke({"input": prompt})['answer']
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

