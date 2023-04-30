import os
from apiKey import apiKey
import streamlit as st
from modules.agent_module import wiki
from modules.memory_module import cast_memory, story_memory
from modules.chains_module import cast_chain, story_chain

os.environ['OPENAI_API_KEY'] = apiKey

# App Structure
st.title('StoryGen')
prompt = st.text_input(
    "Enter a story title")

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
