from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
import torch.nn.functional as F
from app.cnn import SimpleCNN
from PIL import Image
from app.cnn import SimpleCNN
from io import BytesIO


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
classes = ["Angry", "Focused", "Neutral", "Tired"]

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/image")
async def get_prediction(image: UploadFile):
    imageData = await image.read()
    data = await predict(imageData)
    index = data[0]
    return {"emotion": classes[index]}

async def predict(file):
    model = await load_model()
    file_io = BytesIO(file)
    image = Image.open(file_io)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        model.eval()
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        _, predicted_class = torch.max(probabilities, 1)
    return predicted_class.item(), probabilities.squeeze().tolist()

async def load_model():
    model = SimpleCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "app/model/final_model_1.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model