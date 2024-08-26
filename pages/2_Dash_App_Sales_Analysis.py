import streamlit as st
from PIL import Image
import nbformat

st.set_page_config(page_title="Dash Project", page_icon=":mate:", layout="wide")

with st.container():
    st.subheader("Language: Python")
    st.title("Sales Analytics Dash App")
    st.write(
        """
        This project is mean to display my ability to use different libraries to create a developed sales analysis tool using dash. This consists of two parts. One is a notebook for exploring the data while the second initiates the app. 
        """
    )
with st.container():
    st.subheader("Example Output")
img_output = Image.open("images/dash_sample_output.jpeg")
with st.container():
    st.image(img_output)

with st.container():
    st.subheader("Exploritory Analysis")
    st.subheader("Jupiter Notebook")

with open('projects/sales_data_exploritory_data_analysis.ipynb', 'r') as f:
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

with st.container():
    st.subheader("App Developement")
    st.subheader("Python File")

# Read the Python file
with open('projects/sales_dash_app.py', 'r') as file:
    python_code = file.read()

# Display the code in Streamlit with syntax highlighting
st.code(python_code, language='python')