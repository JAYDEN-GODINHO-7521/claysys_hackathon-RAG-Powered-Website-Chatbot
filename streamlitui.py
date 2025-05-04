import streamlit as st
import webscraper as webscr# Assuming you have a function for your RAG logic
import retriver as retrive
st.title(" RAG-Powered Website Chatbot ")

website=st.text_input("Enter the website URL:")

if website:
    with st.spinner("scraping the website..."):
        nos=webscr.main(website)
    st.write(nos)
    


query = st.text_input("Ask me anything about the website:")
if query:
    with st.spinner("Thinking..."):
        response=retrive.main(query)
    st.subheader("Answer:")
    st.write(response)

