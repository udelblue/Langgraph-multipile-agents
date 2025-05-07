import os
import json

import streamlit as st
from agent_graph.graph import create_graph, compile_workflow, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()


class ChatWorkflow:
    def __init__(self):
        self.workflow = None
        self.recursion_limit = 40
        self.graph = None

    def build_workflow(self, server, model, model_endpoint, temperature, recursion_limit=40, stop=None):
        graph = create_graph(
            server=server, 
            model=model, 
            model_endpoint=model_endpoint,
            temperature=temperature,
            stop=stop
        )
        self.graph = graph
        self.workflow = compile_workflow(graph)
        self.recursion_limit = recursion_limit

    def invoke_workflow(self, message):
        if not self.workflow:
            return "Workflow has not been built yet. Please update settings first."
        
        dict_inputs = {"research_question": message.content}
        limit = {"recursion_limit": self.recursion_limit}
        reporter_state = None

        for event in self.workflow.stream(dict_inputs, limit):
            next_agent = ""
            if "router" in event.keys():
                state = event["router"]
                reviewer_state = state['router_response']
                # print("\n\nREVIEWER_STATE:", reviewer_state)
                reviewer_state_dict = json.loads(reviewer_state)
                next_agent_value = reviewer_state_dict["next_agent"]
                if isinstance(next_agent_value, list):
                    next_agent = next_agent_value[-1]
                else:
                    next_agent = next_agent_value

            if next_agent == "final_report":
                # print("\n\nEVENT_DEBUG:", event)
                state = event["router"]
                reporter_state = state['reporter_response']
                if isinstance(reporter_state, list):
                    print("LIST:", "TRUE")
                    reporter_state = reporter_state[-1]
                return reporter_state.content if reporter_state else "No report available"

        return "Workflow did not reach final report"

# Use a single instance of ChatWorkflow
chat_workflow = ChatWorkflow()


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Chatbot")

with st.sidebar:
    st.write("## Langgraph Graph diagram")
    st.write("This is a graph of the chatbot's knowledge and capabilities.")
    graph_bytes = chat_workflow.graph.draw_mermaid_png()
    st.image(graph_bytes, caption="Chatbot Graph")
    


# Display the chat history.
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# Handle user input.
if prompt := st.chat_input("What's on your mind?"):
    # Add user message to chat history
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Create a placeholder for the streaming response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            state = State(messages=[user_message])
            

            # invoke the agent, streaming tokens from any llm calls directly
            for chunk, metadata in graph.stream(state, config={"configurable": {"thread_id": "thread"}}, stream_mode="messages"):
                if isinstance(chunk, AIMessage):
                    full_response = full_response + str(chunk.content)
                    message_placeholder.markdown(full_response + "▌")

                elif isinstance(chunk, ToolMessage):
                    full_response = full_response + f"🛠️ Used tool to get: {chunk.content}\n\n"
                    message_placeholder.markdown(full_response + "▌")

            # Once streaming is complete, display the final message without the cursor
            message_placeholder.markdown(full_response)

            # Add the complete message to session state
            st.session_state.messages.append(AIMessage(content=full_response))
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}") 