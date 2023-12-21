import streamlit as st
import utils
import time
st.title("Chat-with-your-GitHub-Repo")

if True:
    try:
        user_repo = st.text_input("Github Link to your public codebase", key="github_link_input")
    except Exception as e:
        pass
    

    if user_repo:
        st.write("You entered:", user_repo)

        embedder = utils.Embedder(user_repo)
        embedder.clone_repo()
        st.write("Your repo has been cloned")

        st.write("Parsing the content and embedding it. This may take some time")
        embedder.load_db()
        st.write("Done Loading. Ready to take your questions")


        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


        try:
            if prompt := st.text_input("Enter your prompt", key="prompt_input"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                response = embedder.chat_data(prompt)
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            pass
