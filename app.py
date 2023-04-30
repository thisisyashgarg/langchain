import os
from apiKey import apiKey
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apiKey

# Prompt Template
cast_template = PromptTemplate(
    input_variables=['title'],
    template="Write a list of characters with their persona for a story whose title is : {title}"
)
story_template = PromptTemplate(
    input_variables=['cast', 'wikipedia_research'],
    template="Write a story according to the cast which are {cast} while leveraging the wikipedia research as well : {wikipedia_research}"
)

# Memory
cast_memory = ConversationBufferMemory(
    input_key='title', memory_key='chat_history')
story_memory = ConversationBufferMemory(
    input_key='cast', memory_key='chat_history')

# App Structure
st.title('My AutoGPT')
prompt = st.text_input(
    "Enter a story title")

# LLM
llm = OpenAI(temperature=0.75)
cast_chain = LLMChain(llm=llm, prompt=cast_template,
                      verbose=True, output_key='cast', memory=cast_memory)
story_chain = LLMChain(llm=llm, prompt=story_template,
                       verbose=True, output_key='story', memory=story_memory)
wiki = WikipediaAPIWrapper()
# sequential_chain = SequentialChain(
#     chains=[cast_chain, story_chain], input_variables=['title'], output_variables=['cast', 'story'], verbose=True)

# Response from the AI
if prompt:
    # response = sequential_chain({"title": prompt})
    cast = cast_chain.run(prompt)
    wiki_research = wiki.run(cast)
    story = story_chain.run(cast=cast, wikipedia_research=wiki_research)

    st.write(cast)
    st.write(story)

    # History Chats
    with st.expander("Cast History"):
        st.info(cast_memory.buffer)

    with st.expander("Story History"):
        st.info(story_memory.buffer)

    with st.expander("Wikipedia History"):
        st.info(wiki_research)
