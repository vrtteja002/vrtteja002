# Main Portfolio Repository

## README.md

```markdown
# Ravi Teja Vempati - AI/ML Portfolio

## About Me
I'm an AI/ML Engineer with a strong background in computer vision, natural language processing, and deep learning. Currently pursuing my Master's in Computer Science at Kansas State University, I'm passionate about developing innovative AI solutions that solve real-world problems.

## Education
- **Master of Science in Computer Science**
  Kansas State University, Manhattan, Kansas (Expected Dec 2024)
- **Bachelor of Technology in Computer Science**
  Vellore Institute of Technology, Chennai, India (Aug 2022)

## Technical Skills
- **Languages**: Python, C, C++
- **ML/DL Frameworks**: PyTorch, TensorFlow, Scikit-Learn
- **AI/NLP**: LangChain, Large Language Models, CLIP, Generative AI
- **Computer Vision**: OpenCV, YOLOv8, YOLOv9
- **Cloud**: AWS (SageMaker, Bedrock), Google Cloud, Azure

## Work Experience
- **AI/ML Engineer Intern** - Radical AI, New York, NY (Jun 2024 - Present)
- **AI/ML Engineer Fellow** - Fellowship.AI, San Francisco, CA (Apr 2024 - Jun 2024)
- **Graduate Research Assistant** - Kansas State University (May 2023 - Present)

## Project Highlights
1. [Apple Leaf Disease Classification](link-to-repo)
2. [ASL and Facial Expression Recognition](link-to-repo)
3. [Vehicle Detection and Counting](link-to-repo)
4. [Gender Classification Using Facial Recognition](link-to-repo)
5. [Real Estate Agent App with LLM Integration](link-to-repo)
6. [Quizify AI Feature](link-to-repo)

## Certifications
- OCI Generative-AI Professional, Oracle (2024)
- Multi-Agent-Systems (CrewAI), DeepLearning.AI (2024)
- LangChain for LLM Application Development, DeepLearning.AI (2024)
- Machine Learning Using Python, Vellore Institute of Technology (2022)

## Contact
- Email: vrtteja001@ksu.edu
- LinkedIn: [Ravi Teja Vempati](https://www.linkedin.com/in/ravi-teja-vempati)
- GitHub: [YourGitHubUsername](https://github.com/YourGitHubUsername)
```

# Project Repository: Apple Leaf Disease Classification

## README.md

```markdown
# Apple Leaf Disease Classification

This project implements deep learning models for classifying apple leaf diseases using transfer learning with ResNet-50 and a custom CNN architecture.

## Overview
The goal of this project is to accurately classify apple leaf diseases, which can help in early detection and treatment of plant diseases in agriculture. We've implemented two models:
1. Transfer Learning with ResNet-50
2. Custom CNN Architecture

## Results
- ResNet-50: 98.54% accuracy
- Custom CNN: 98.33% accuracy

## Technologies Used
- Python
- PyTorch
- torchvision
- numpy
- matplotlib

## Setup and Installation
1. Clone this repository:
   ```
   git clone https://github.com/yourusername/apple-leaf-disease-classification.git
   cd apple-leaf-disease-classification
   ```
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Prepare your dataset in the following structure:
   ```
   data/
   ├── train/
   │   ├── healthy/
   │   ├── scab/
   │   ├── rust/
   │   └── multiple_diseases/
   └── test/
       ├── healthy/
       ├── scab/
       ├── rust/
       └── multiple_diseases/
   ```
2. Run the training script:
   ```
   python train.py --model resnet50  # or --model custom_cnn
   ```
3. For inference on new images:
   ```
   python predict.py --image path/to/your/image.jpg --model resnet50
   ```

## Model Architecture
[Include details about the ResNet-50 architecture and your custom CNN architecture here]

## Performance Analysis
[Include graphs or tables comparing the performance of both models]

## Future Improvements
- Experiment with data augmentation techniques
- Try other pre-trained models like EfficientNet or Vision Transformer
- Implement an ensemble of multiple models for potentially higher accuracy

## Contributors
- Ravi Teja Vempati

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

## train.py

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import argparse

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load data
train_data = datasets.ImageFolder('data/train', transform=transform)
test_data = datasets.ImageFolder('data/test', transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Define models
def get_resnet50():
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)  # 4 classes
    return model

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = self.dropout(nn.functional.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Training function
def train_model(model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

    print('Training complete')

# Evaluation function
def evaluate_model(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train apple leaf disease classification model')
    parser.add_argument('--model', type=str, choices=['resnet50', 'custom_cnn'], required=True)
    args = parser.parse_args()

    if args.model == 'resnet50':
        model = get_resnet50()
    else:
        model = CustomCNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, criterion, optimizer)
    evaluate_model(model)

    # Save the model
    torch.save(model.state_dict(), f'{args.model}_apple_disease.pth')
```

## predict.py

```python
import torch
from torchvision import transforms
from PIL import Image
import argparse
from train import get_resnet50, CustomCNN

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class labels
class_labels = ['healthy', 'multiple_diseases', 'rust', 'scab']

def predict_image(image_path, model):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        
    return class_labels[predicted.item()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict apple leaf disease')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--model', type=str, choices=['resnet50', 'custom_cnn'], required=True)
    args = parser.parse_args()

    # Load the trained model
    if args.model == 'resnet50':
        model = get_resnet50()
    else:
        model = CustomCNN()
    
    model.load_state_dict(torch.load(f'{args.model}_apple_disease.pth'))
    model.eval()

    prediction = predict_image(args.image, model)
    print(f'The predicted class is: {prediction}')
```
