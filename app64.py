from flask import Flask, flash, request, redirect, url_for, render_template
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
sys.path.insert(1, "C:\\Users\\Vincent\\OneDrive - UTS\\5 Year\\Honors Project\\DeepHash-pytorch-master")
from Unsupervised_GreedyHash import GreedyHashModelUnsupervised
from werkzeug.utils import secure_filename
import time as tm

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

img_dir = "C:\\Users\\Vincent\\OneDrive - UTS\\5 Year\\Honors Project\\Thumbnails\\"
with open("C:\\Users\\Vincent\\OneDrive - UTS\\5 Year\\Honors Project\\Hashing\\filename64.txt", "r") as f:
    trn_img_path = np.array([img_dir + item.split("\n")[0] + '.png' for item in f.readlines()])
save_path = "C:\\Users\\Vincent\\OneDrive - UTS\\5 Year\\Honors Project\\DeepHash-pytorch-master\\Save_Path\\coco_64bits_0.7031763260301034\\"
#trn_binary = np.load(save_path + "trn_binary.npy")
hash_path="C:\\Users\\Vincent\\OneDrive - UTS\\5 Year\\Honors Project\\Hashing\\"
offset_table = np.genfromtxt(hash_path + "offset_table64.txt", dtype='int')
#print (offset_table)
trn_binary = np.genfromtxt(hash_path + "database64.txt")
#print (trn_binary)
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

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                                ])


# Enter the path, return the hash code
def detect(source):
    img = Image.open(source).convert('RGB') # w x h x 3
    # print(np.asarray(img).shape)
    img = transform(img)
    img =img.unsqueeze(0) # 1 x 3 x 224 x 224
    #try to split the transform and unsqueeze
    qB = model(img).sign()[0].detach().numpy() # 1 x 64
    #print (qB)
    return qB


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    
    return distH


def retrival(qB, start=0, end=50):
    # Calculate Hamming distance by hash code
    hamm = CalcHammingDist(qB, trn_binary)
    # Calculate the indices of the nearest n distances (how they would sort closest to the image given)
    ind = np.argsort(hamm)[start:end] #(gota filter ind into 1101 shape)
    #test = ind[0] - offset_table[0]
    print (ind)
    #print (offset_table[0])
    #print (test)
    i = 0
    inde = []
    result_time = []
    final_index = []
    k=0
    while i < len(ind):
        j = 0
        while j < len(offset_table):
            if ind[i]< offset_table[j]:
                inde.append(j-1)
                for k in inde:
                    if k not in final_index:
                        final_index.append(k)
                        time_frame = 0
                        time_frame = ([ind[i] - offset_table[k]])
                        time_frame = (sum(time_frame) // 4)
                        time_frame = tm.strftime('%M:%S', tm.gmtime(time_frame))
                        result_time.append(time_frame)
                break
            j += 1
        i += 1
    #print (result_time)
    #print (final_index)
    # Returns the true value of the result
    # Returns the Hamming distance of the result
    result_hamm = hamm[final_index].astype(int)
    #print (result_hamm)
    result_path = trn_img_path[final_index]
    result_code = trn_binary[final_index]
    result = []
    for hmm, path, code, time in zip(result_hamm, result_path, result_code, result_time):
        row = {}
        row["hmm"] = int(hmm)
        with open(path, 'rb') as img_f:
            img_stream = img_f.read()
            img_stream = base64.b64encode(img_stream).decode()
        #row["img"] = img_stream
        #try this ^ but with the name of file
        #img_stream = 'test.png'
        
        row["img"] = img_stream
        row["code"] = convert0(code)
        name = path[65:]
        size = len(name)
        name = name[:size - 4]
        row["path"] = name
        row["time"] = time
        #print (name)
        result.append(row)
        #print (result[0])
    return result


# String +1, -1 -> 01
def convert0(code):
    return "".join(code.astype(int).astype(str).tolist()).replace("-1", "0")


def convert1(code):
    code = list(code)
    code = [-1.0 if (c == "0") else 1.0 for c in code]
    return np.array(code)

app = Flask(__name__)
app.secret_key = "secret key"



@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    f = request.files['file']
    f.save("areyouok.png")
    qB = detect("areyouok.png")
    qB_binary = convert0(qB)
    print(f)
    result = retrival(qB, end=50)
    response = {
        "qB": qB_binary,
        "result": result
    }
    return json.dumps(response, ensure_ascii=False)

@app.route('/video')
def upload_form():
    filename = request.args.get('filename')
    filename = secure_filename(filename)
    filename = filename + '.mp4'
    print (filename)
    return render_template("video.html", filename=filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
