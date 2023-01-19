from flask import Flask, render_template, request
import json
import base64
import warnings
import torch
warnings.filterwarnings("ignore")
import os
from PIL import Image
import numpy as np
import torch.nn as nn
from torchvision import models
import sys
sys.path.insert(1, '/data/vhngai/DeepHash-pytorch-master')
from Unsupervised_GreedyHash import GreedyHashModelUnsupervised


class ResNet(nn.Module):
    def __init__(self, hash_bit):
        super(ResNet, self).__init__()
        model_resnet = models.resnet50(pretrained=False)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)
        return y


# Just use it~
device = torch.device('cpu')
from torchvision import transforms

img_dir = "/data/vhngai/DeepHash-pytorch-master/dataset/COCO_2014/"
with open("/data/vhngai/DeepHash-pytorch-master/data/coco/database.txt", "r") as f:
    trn_img_path = np.array([img_dir + item.split(" ")[0] for item in f.readlines()])
save_path = "/data/vhngai/DeepHash-pytorch-master/Save_Path/coco_64bits_0.7031763260301034/"
trn_binary = np.load(save_path + "trn_binary.npy")
# # load model
print("loading the model。。。。。。。")
# Write the model path here
model_name = 'model.pt'
model_state_dict = torch.load(save_path + model_name, map_location=device)
# Hash code length 64
model = GreedyHashModelUnsupervised(64)
model.load_state_dict(model_state_dict)
model.eval()
print("Model loaded successfully")

transform = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                                ])


# Enter the path, return the hash code
def detect(source):
    img = Image.open(source).convert('RGB') # w x h x 3
    print(np.asarray(img).shape)
    img = transform(img).unsqueeze(0) # 1 x 3 x 224 x 224

    '''
    video = []
    for frame in frames:
        f = transform(frame)
        video.append(f)
    video = np.asarray(video)
    qB = model(video).sign()[0].detach().numpy()
    print(qB)
    print(qB.shape)
    '''

    qB = model(img).sign()[0].detach().numpy() # 1 x 64

    return qB


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def retrival(qB, start=0, end=50):
    # Calculate Hamming distance by hash code
    hamm = CalcHammingDist(qB, trn_binary)
    # Calculate the indices of the nearest n distances
    ind = np.argsort(hamm)[start:end]
    # Returns the true value of the result
    # Returns the Hamming distance of the result
    result_hamm = hamm[ind].astype(int)
    result_path = trn_img_path[ind]
    result_code = trn_binary[ind]
    result = []
    for hmm, path, code in zip(result_hamm, result_path, result_code):
        row = {}
        row["hmm"] = int(hmm)
        with open(path, 'rb') as img_f:
            img_stream = img_f.read()
            img_stream = base64.b64encode(img_stream).decode()
        row["img"] = img_stream
        row["code"] = convert0(code)
        result.append(row)
    return result


# String +1, -1 -> 01
def convert0(code):
    return "".join(code.astype(int).astype(str).tolist()).replace("-1", "0")


def convert1(code):
    code = list(code)
    code = [-1.0 if (c == "0") else 1.0 for c in code]
    return np.array(code)


app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    f = request.files['file']
    f.save("areyouok.png")
    qB = detect("areyouok.png")
    qB_binary = convert0(qB)
    # print(qB_binary)
    result = retrival(qB, end=50)
    response = {
        "qB": qB_binary,
        "result": result
    }
    # print(response)
    return json.dumps(response, ensure_ascii=False)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
