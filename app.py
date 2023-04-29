import os
from apiKey import apiKey
import streamlit as st
from langchain.llms import OpenAI

os.environ['OPENAI_API_KEY'] = apiKey

# App
st.title('My AutoGPT')
prompt = st.text_input("Type your input")

# llms
llm = OpenAI(temperature=0.75)

# response
if prompt:
    response = llm(prompt)
    st.write(response)
