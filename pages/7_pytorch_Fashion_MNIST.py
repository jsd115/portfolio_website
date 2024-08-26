import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import nbformat

st.set_page_config(page_title="Pytoch Clothing Classification", page_icon=":mate:", layout="wide")

with st.container():
    st.subheader("Language: Python")
    st.title("Clothing Classification with Pytorch")
    st.write(
        """
        This model is trained on the Fashion MNIST dataset. The model is a convolutional neural network (CNN) that is programmed using the Pytorch library. The model is limited due to computational power, but could be improved with more layers and more training epochs. 
        """
    )

# Load the trained model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 11 * 11, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128 * 11 * 11)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CNN()
model.load_state_dict(torch.load('projects/model.pth'))
model.eval()

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((28, 28)),  # Resize to 28x28
        transforms.ToTensor(),  # Convert to tensor
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Define a function to make predictions
def predict(image):
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

with st.container():
    st.subheader("Model Demo")
    st.write("##")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess the image
        image = preprocess_image(image)
        
        # Make prediction
        prediction = predict(image)

        # Associate the prediction with the class name
        classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        # Display the prediction
        st.write(f"Predicted class: {prediction}, Class name: {classes[prediction]}")
    st.write("---")
    st.subheader("Project Code")



with open('projects/pytorch_portfolio_project.ipynb', 'r') as f:
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