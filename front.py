import streamlit as st
import requests

URL = "http://localhost:8000/predict/{}"

def main():
    text_input = st.text_area("Escreva uma frase:")
    if text_input != "":
        url = URL.format(text_input)
        res = requests.get(url)
        output = '\n\n'.join(res.json()['sentiments'])
        st.markdown(output)

if __name__ == '__main__':
    main()
    