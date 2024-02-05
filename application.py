import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import warnings
warnings.filterwarnings("ignore")
import re
import pandas as pd
import numpy as np
import re 
import random
import base64
from io import BytesIO

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import requests

import warnings
warnings.filterwarnings("ignore")

#loading Dataset
df = pd.read_csv('dataset/Songs_Dataset_Final.csv',encoding='utf-8')

application = Flask(__name__)
app = application

## Import word embedding model

class BERT_Arch(nn.Module):

    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 6)  # Adjust the number of output neurons for 123 classes
        self.log_softmax = nn.LogSoftmax(dim=1)  # Use dim=1 for LogSoftmax in classification

    def forward(self, sent_id, mask):
        sent_id = torch.tensor(sent_id)
        outputs = self.bert(input_ids=sent_id, attention_mask=mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        x = self.fc1(last_hidden_state_cls)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x


def get_output(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    my_dict = {
            0:'Sad', 1:'Happy', 2:'Happy', 3: 'Sad', 4: 'Fear', 5:'Suprise'
    }
    
    def NLP_cleaning(text):
        sent = text
        sent = re.sub('<[^>]*>', '', sent)
        sent = re.sub('[^a-zA-z0-9]', ' ', sent)
        sent = sent.lower()
        return sent
    
    def load_checkpoint(filepath):
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
        return model
    
    model = load_checkpoint('model/checkpoint.pth')
    device = 'cpu'
    
    def Predict(text):
        encoded_review = tokenizer.encode_plus(
          text,
          max_length=256,
          truncation = True,
          add_special_tokens=True,
          return_token_type_ids=False,
          padding = 'max_length',
          return_attention_mask=True,
          return_tensors='pt',
        )

        input_ids = encoded_review['input_ids'].to(device)
        attention_mask = encoded_review['attention_mask'].to(device)
        output = model(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)
        extracted_value = prediction.item()
        return extracted_value
    
    preprocessed = NLP_cleaning(text)
    output = Predict(preprocessed)
    return my_dict[output]

def ChooseDataset(x):
    if x == "Fear":
        return df[df['Mood'].isin(['Happy', 'Calm'])]
    if x == "Happy":
        return df[df['Mood'].isin(['Energetic', 'Happy', 'Calm'])]
    if x == "Sad":
        return df[df['Mood'].isin(['Sad', 'Calm'])]
    if x == "Surprise":
        return df[df['Mood'].isin(['Calm'])]
    return df

## Home page Route
@app.route('/')
def index():
    return render_template('index.html')

## Forms page
@app.route('/predictdata' , methods = ['GET','POST'])
def predict_datapoints():
    if request.method == 'POST':
        Text = str(request.form.get('Text'))
        string = get_output(Text)

        dataset = ChooseDataset(string)
        random_songs = dataset.sample(n=20)

        songs_info = []
        client_id = 'bdb135bd934845888ebbca4503dafa52'
        client_secret = '1c4104ba0ef147a09e198023b3d340a4'

        for index, song in random_songs.iterrows():
            song_info = {
                'name': song['name'],
                'spotify_url': f"https://open.spotify.com/track/{song['id']}"
            }
            songs_info.append(song_info)
    
            url = f"https://api.spotify.com/v1/tracks/{song['id']}"
            client_credentials = f"{client_id}:{client_secret}"
            encoded_credentials = base64.b64encode(client_credentials.encode()).decode()
            headers = {
                "Authorization": f"Basic {encoded_credentials}"
            }
            data = {
                "grant_type": "client_credentials"
            }
            token_url = "https://accounts.spotify.com/api/token"
            token_response = requests.post(token_url, headers=headers, data=data)
            if token_response.status_code == 200:
                access_token = token_response.json()['access_token']
                headers = {
                    "Authorization": f"Bearer {access_token}"
                }
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    image_url = data['album']['images'][0]['url']
                    song_info['image_url'] = image_url


        return render_template('recommendations.html', songs_info=songs_info)

        

           
    
    else:
        return render_template('recommendations.html')
    
if __name__ == "__main__":
    app.run(host = "0.0.0.0")

    




