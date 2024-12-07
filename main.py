from flask import Flask, request, jsonify
from torchvision import transforms
import torch.nn.functional as F
from cnn import SimpleCNN
from PIL import Image
from flask_cors import CORS,cross_origin
import torch
app = Flask(__name__)
cors = CORS(app)

@app.route("/")
def hello_world():
    return "Hello world"

@app.route("/image", methods=["POST"])
@cross_origin()
def get_results():
    file = request.files['image']
    print(file)
    results = predict_image(file)
    classes = ['Angry', 'Focused', 'Neutral', 'Tired']
    predicted_class = classes[results[0]]
    predicted_probabilities = results[1]
    final_results = {}
    for i in range(len(classes)):
        final_results[classes[i]] = predicted_probabilities[i]
    return jsonify(predicted_class, final_results)


def predict_image(file):
    model = load_model()
    image = Image.open(file)
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

def load_model():
    model = SimpleCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./model/final_model_1.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model