import pandas as pd
import numpy as np
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, render_template, request, jsonify
from torch.utils.data import DataLoader, TensorDataset
import os
import sys


# Initialize Flask app
app = Flask(__name__)
project_home = '/home/sana-a-p/fake-review-detection'
project_home = '/home/yourusername/yourproject'
if project_home not in sys.path:
    sys.path.insert(0, project_home)
# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset
data = pd.read_csv('yelp.csv')

# Preprocess text function
def preprocess_text(text):
    tokens = word_tokenize(str(text).lower())  # Convert to lowercase & tokenize
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]  # Remove stopwords & non-alpha
    return tokens

data['cleaned_reviews'] = data['text'].apply(preprocess_text)

# Train or load Word2Vec model
W2V_PATH = "w2v_model.bin"
if os.path.exists(W2V_PATH):
    w2v_model = Word2Vec.load(W2V_PATH)
else:
    w2v_model = Word2Vec(sentences=data['cleaned_reviews'].tolist(), vector_size=300, window=5, min_count=1, workers=4)
    w2v_model.save(W2V_PATH)

# Convert reviews into Word2Vec embeddings
def review_to_vector(review, model, vector_size=300):
    vectors = [model.wv[word] for word in review if word in model.wv]
    if not vectors:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

X = np.array([review_to_vector(review, w2v_model) for review in data['cleaned_reviews']])
y = data['validity'].values  # Labels column

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).float().view(X.shape[0], 1, X.shape[1])  # Reshape for LSTM
y_tensor = torch.from_numpy(y).float().unsqueeze(1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

# Define MIANA model
class MIANA(nn.Module):
    def __init__(self, input_size=300, hidden_size=150):
        super(MIANA, self).__init__()
        self.bi_lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.bi_lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        weighted_sum = torch.sum(attention_weights * lstm_out, dim=1)
        output = self.fc(weighted_sum)
        return self.sigmoid(output)

# Train or load model
MODEL_PATH = "miana_model.pth"
model = MIANA(input_size=300, hidden_size=150)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Model loaded successfully.")
else:
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(5):  # Train for 5 epochs
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze(1)
            loss = criterion(predictions, y_batch.squeeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation after each epoch
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                predictions = model(X_batch).squeeze(1)
                val_loss += criterion(predictions, y_batch.squeeze(1)).item()
                correct += ((predictions > 0.5).float() == y_batch.squeeze(1)).sum().item()
                total += y_batch.size(0)
        
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Validation Loss: {val_loss:.4f}, Accuracy: {correct/total:.2%}")
    
    torch.save(model.state_dict(), MODEL_PATH)

# Flask API for predictions
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        reviews = data.get('reviews', [])
        
        if not reviews:
            return jsonify({'error': 'No reviews provided'}), 400
        
        processed_reviews = [preprocess_text(review) for review in reviews]
        vectors = np.array([review_to_vector(tokens, w2v_model) for tokens in processed_reviews])
        tensor_input = torch.tensor(vectors, dtype=torch.float32)
        
        with torch.no_grad():
            probabilities = model(tensor_input).squeeze(1).tolist()
        
        return jsonify({'fake_review_probabilities': probabilities})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run Flask server
if __name__ == '__main__':
    app.run(debug=True)
