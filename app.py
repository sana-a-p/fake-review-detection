from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__)

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load trained Word2Vec model
w2v_model = Word2Vec.load("word2vec.model")

# Define MIANA model
class MIANA(nn.Module):
    def __init__(self, input_size=300, hidden_size=150):
        super(MIANA, self).__init__()
        self.bi_lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.bi_lstm(x.unsqueeze(1))  # Add sequence dimension
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        weighted_sum = torch.sum(attention_weights * lstm_out, dim=1)
        output = self.fc(weighted_sum)
        return self.sigmoid(output)

# Load trained model
model = MIANA()
model.load_state_dict(torch.load("miana_model.pth"))
model.eval()

# Preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

# Convert review text into embeddings
def review_to_vector(review, model, vector_size=300):
    vectors = [model.wv[word] for word in review if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

# Serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Predict API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        review_text = data.get('review', '')

        # Preprocess input review
        tokens = preprocess_text(review_text)
        vector = review_to_vector(tokens, w2v_model)
        tensor_input = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)

        # Predict using model
        with torch.no_grad():
            probability = model(tensor_input).item()

        return jsonify({'fake_review_probability': probability})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

