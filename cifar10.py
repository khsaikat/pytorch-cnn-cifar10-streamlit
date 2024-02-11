import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision
import torchvision.transforms as transforms
import streamlit as st
# import captum
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class CNN(nn.Module):
    def __init__(self, k: int):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, k)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = f.dropout(x, p=0.5)
        x = f.relu(self.fc1(x))
        x = f.dropout(x, p=0.2)
        x = self.fc2(x)
        return x


@st.cache_resource
def load_model(processor_device):
    trained_model = CNN(10)
    trained_model.to(processor_device)
    trained_model.load_state_dict(torch.load('./models/cifar10-cnn.pt'))
    categories = '''airplanes cars birds cats deer dogs frogs horses ships trucks'''.split()
    trained_model.eval()
    return categories, trained_model


# Make prediction
def make_prediction(trained_model, processed_input):

    with torch.no_grad():
        output = trained_model(processed_input.to(device))
        pred = output.softmax(1)
        pred = pred[0].detach().cpu().numpy()
        probs, idxs = pred[pred.argsort()[-5:][::-1]], pred.argsort()[-5:][::-1]
        print(idxs)
        return probs, idxs
    # Get predicted class
    # _, predicted_class = torch.max(output, 1)
    # print(f"Predicted class index: {predicted_class.item()}")


# Streamlit Dashboard GUI

st.title("CIFAR-10 PyTorch Pretrained Image Classifier :coffee:")
upload = st.file_uploader(label="Upload Image:", type=["png", "jpg", "jpeg"])

if upload:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on: " + str(device))
    categories, model = load_model(device)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # image_path = './test_images/test_image.jpg'
    image = Image.open(upload)
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    probs, idxs = make_prediction(model, input_batch)

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        main_fig = plt.figure(figsize=(6, 6))
        ax = main_fig.add_subplot(111)
        plt.imshow(image)
        plt.xticks([], [])
        plt.yticks([], [])
        st.pyplot(main_fig, use_container_width=True)

    with col2:
        main_fig = plt.figure(figsize=(6,3))
        ax = main_fig.add_subplot(111)
        plt.bar(x=idxs, height=probs, color="tomato")
        plt.title("Top 5 Probabilities", loc="center", fontsize=15)
        st.pyplot(main_fig, use_container_width=True)
