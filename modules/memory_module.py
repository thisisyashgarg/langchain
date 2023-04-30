from langchain.memory import ConversationBufferMemory

# Memory
cast_memory = ConversationBufferMemory(
    input_key='title', memory_key='chat_history')
story_memory = ConversationBufferMemory(
    input_key='cast', memory_key='chat_history')
