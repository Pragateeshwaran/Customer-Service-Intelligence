import streamlit as st
import os
from main import csi
from streamlit_chat import message
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

if "llm" not in st.session_state:
    st.session_state.llm = ChatGroq(model_name="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))
if "result" not in st.session_state:
    st.session_state.result = None
if "transcripts" not in st.session_state:
    st.session_state.transcripts = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conmessages" not in st.session_state:
    st.session_state.conmessages = []

st.markdown("""
<style>
.st-emotion-cache-13k62yr {
    background-image: url('https://firebasestorage.googleapis.com/.../black_needles.jpg');
    background-size: cover;
}
.st-emotion-cache-1cypcdb,
.st-emotion-cache-4rht51,
.st-emotion-cache-1erivf3,
.st-emotion-cache-1avcm0n {
    background: rgb(0 0 0);
}
</style>
""", unsafe_allow_html=True)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Upload audio before going to the chat menu.", ["Upload Audio", "Chat"])
    st.write("Team Quixotic Sapiens")
    st.title("Customer service Analysis")

    if page == "Upload Audio":
        uploaded_file = st.file_uploader("Upload your audio file here:", type=["wav", "mp3"])
        if uploaded_file:
            st.write("File uploaded successfully!..Wait for the report..")
            with open("temp.wav", "wb") as f:
                f.write(uploaded_file.getvalue())
            app = csi('llama3-70b-8192', st.session_state.llm)
            st.session_state.result, st.session_state.transcripts = app.process_return_with_transcripts("temp.wav")
            st.write("Processed Result:")
            st.write(st.session_state.result)

    if page == "Chat":
        st.session_state.messages = [
            f"### System:You are a customer service expert...\n### User:{st.session_state.transcripts}\n### Assistant:{st.session_state.result}"
        ]
        user_input = st.text_input("Your message: ", key="user_input")
        if user_input:
            st.session_state.conmessages.append(user_input)
            st.session_state.messages.append(f"\n### User:\n{user_input}\n### Assistant:\n")
            prompt_template = PromptTemplate(input_variables=["messages"], template="{messages}")
            chain = prompt_template | st.session_state.llm
            response = chain.invoke({"messages": " ".join(st.session_state.messages)})
            reply = response.content.split("### Assistant:\n")[-1]
            st.session_state.conmessages.append(reply)
            for i, msg in enumerate(st.session_state.conmessages):
                message(msg, is_user=(i % 2 != 0), key=str(i))

if __name__ == "__main__":
    main()
