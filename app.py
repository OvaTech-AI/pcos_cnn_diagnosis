import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import gradio as gr

# Define the CNN model architecture
class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 12, kernel_size=5, padding=2)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(12, 8, kernel_size=5, padding=2)
        self.conv3 = torch.nn.Conv2d(8, 4, kernel_size=5, padding=2)
        self._to_linear = None
        self.convs(torch.randn(1, 3, 224, 224))
        self.fc1 = torch.nn.Linear(self._to_linear, 1)

    def convs(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        if self._to_linear is None:
            self._to_linear = x.view(-1).size(0)
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = self.fc1(x)
        return x

# Load the model
model = CNNModel()
model.load_state_dict(torch.load('pcos_cnn_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define the image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define the prediction function
def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output).item()
    print(f"Raw output: {output}, Sigmoid output: {prediction}")
    return "Infected" if prediction < 0.5 else "Not Infected"

# Define the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(),
    title="PCOS Diagnosis",
    description="Upload an ultrasound image of the ovary to get a prediction about PCOS."
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch()
