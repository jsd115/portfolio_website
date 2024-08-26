import streamlit as st
from PIL import Image
import nbformat

st.set_page_config(page_title="Bankruptcy Classification with TensorFlow", page_icon=":mate:", layout="wide")

with st.container():
    st.subheader("Language: Python")
    st.title("Bankruptcy Classification with TensorFlow")
    st.write(
        """
        This model is trained off of a popular Kaggle dataset and was created to demonstrate a classification model using TensorFlow. This project was a great way to gain hands on experience with TensorFlow packages and to learn how to create a classification model with a high accuracy neural network with real world application. The model additionally combines the utility of sklearn in model selection and hyperparameter tuning to output a very accurate model.
        """
    )

with open('projects/bankruptcy_classification.ipynb', 'r') as f:
    notebook = nbformat.read(f, as_version=4)

# Display code cells and outputs
for cell in notebook.cells:
    if cell.cell_type == 'code':
        st.code(cell.source, language='python')
        if cell.outputs:
            for output in cell.outputs:
                if output.output_type == 'stream':
                    st.text(output.text)
                elif output.output_type == 'display_data':
                    if 'text/plain' in output.data:
                        st.text(output.data['text/plain'])
                    if 'text/html' in output.data:
                        st.markdown(output.data['text/html'], unsafe_allow_html=True)
                elif output.output_type == 'error':
                    st.error(output.evalue)