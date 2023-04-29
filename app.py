import os
from apiKey import apiKey
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

os.environ['OPENAI_API_KEY'] = apiKey

# prompt  template
cast_template = PromptTemplate(
    input_variables=['title'],
    template="Write a list of characters with their persona for a story whose title is : {title}"
)
story_template = PromptTemplate(
    input_variables=['cast'],
    template="Write a story according to the cast which are {cast}"
)

# Memory
memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# App
st.title('My AutoGPT')
prompt = st.text_input(
    "Enter the title of the story and I will give you the story")

# LLM
llm = OpenAI(temperature=0.75)
cast_chain = LLMChain(llm=llm, prompt=cast_template,
                      verbose=True, output_key='cast', memory=memory)
story_chain = LLMChain(llm=llm, prompt=story_template,
                       verbose=True, output_key='story', memory=memory)
sequential_chain = SequentialChain(
    chains=[cast_chain, story_chain], input_variables=['title'], output_variables=['cast', 'story'], verbose=True)

# response
if prompt:
    response = sequential_chain({"title": prompt})
    st.write(response['cast'])
    st.write(response['story'])

    with st.expander("Message History"):
        st.info(memory.buffer)
