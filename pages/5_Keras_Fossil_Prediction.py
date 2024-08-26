import streamlit as st
from PIL import Image
import nbformat

st.set_page_config(page_title="Fossil Age Prediction with Keras", page_icon=":mate:", layout="wide")

with st.container():
    st.subheader("Language: Python")
    st.title("Fossil Age Prediction with Keras")
    st.write(
        """
        This model is trained off of a popular Kaggle dataset and was created to demonstrate my knowledge of the XGBoost library and its applications. Additionally, this project demonstrates the benefits of hyperparameter tuning and the powerful combination of XGBoost and Sklearn together. 
        """
    )

with open('projects/fossil_prediction.ipynb', 'r') as f:
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