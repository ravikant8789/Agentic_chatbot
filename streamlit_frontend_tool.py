import streamlit as st
from langgraph_tool_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command
import uuid

# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

def get_pending_interrupt(thread_id):
    """Retrieve any active interrupt payload from the graph state."""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    if state.next and state.tasks:
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                return task.interrupts[0].value
    return None

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

# ============================ Sidebar ============================
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

st.sidebar.header("My Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id), key=f"thread_{thread_id}"):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_messages
        st.rerun()

# ============================ Main UI ============================

# Render message history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

CONFIG = {
    "configurable": {"thread_id": st.session_state["thread_id"]},
    "metadata": {"thread_id": st.session_state["thread_id"]},
    "run_name": "chat_turn",
}

# ==================== Interrupt Handling (HITL) ====================
pending_prompt = get_pending_interrupt(st.session_state["thread_id"])

if pending_prompt:
    st.warning(f"🔔 **Human Approval Required:** {pending_prompt}")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Approve (Yes)", key="hitl_approve"):
            with st.chat_message("assistant"):
                st.info("Resuming tool execution with approval...")
                resumed_result = chatbot.invoke(Command(resume="yes"), config=CONFIG)
                
                # Update UI history with latest response
                messages = resumed_result.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    st.text(last_msg.content)
                    st.session_state["message_history"].append(
                        {"role": "assistant", "content": last_msg.content}
                    )
            st.rerun()

    with col2:
        if st.button("❌ Decline (No)", key="hitl_decline"):
            with st.chat_message("assistant"):
                st.info("Resuming tool execution with decline...")
                resumed_result = chatbot.invoke(Command(resume="no"), config=CONFIG)
                
                messages = resumed_result.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    st.text(last_msg.content)
                    st.session_state["message_history"].append(
                        {"role": "assistant", "content": last_msg.content}
                    )
            st.rerun()

# ============================ Chat Input ============================
# Disable standard chat input if an interrupt is waiting for decision
user_input = st.chat_input("Type here", disabled=bool(pending_prompt))

if user_input:
    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # Assistant streaming block
    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    if ai_message:
        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_message}
        )

    # Re-run after processing turn to check if execution paused at an interrupt
    st.rerun()
