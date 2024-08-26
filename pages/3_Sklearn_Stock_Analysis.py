import streamlit as st
from PIL import Image
import nbformat

st.set_page_config(page_title="Sklearn Stock Prediction", page_icon=":mate:", layout="wide")

with st.container():
    st.subheader("Language: Python")
    st.title("Stock Prediction App")
    st.write(
        """
        ******* Disclaimer ******* This app is not meant to be used as as professional finance tool!!!! It is simply a passion project meant to display the use of a common python machine learning framework: Sklearn. Nothing in this app is meant to be considered true financial advice. As always, be sure to execute your own financial research as I am not a true financial professional.
        """
    )

with open('projects/stockPricePredictor.ipynb', 'r') as f:
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