import pandas as pd
import numpy as np
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, render_template, request, jsonify
from torch.utils.data import DataLoader, TensorDataset
import os

# Initialize Flask app
app = Flask(__name__)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load dataset
data = pd.read_csv('Home_and_Kitchen.csv')

# Ensure binary labels
data['validity'] = data['validity'].astype(int)

# Preprocess text function
def preprocess_text(text):
    tokens = word_tokenize(str(text).lower())  
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]  
    return tokens

data['cleaned_reviews'] = data['text_'].apply(preprocess_text)

# Stratified train-test split to handle imbalance
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    data['cleaned_reviews'], data['validity'], test_size=0.2, stratify=data['validity'], random_state=42
)

# Train Word2Vec
W2V_PATH = "w2v_model.bin"
if os.path.exists(W2V_PATH):
    w2v_model = Word2Vec.load(W2V_PATH)
else:
    w2v_model = Word2Vec(sentences=X_train_texts, vector_size=300, window=5, min_count=2, workers=4, sg=1, epochs=10)
    w2v_model.save(W2V_PATH)

# Convert reviews into Word2Vec embeddings
def review_to_vector(review, model, vector_size=300):
    vectors = [model.wv[word] for word in review if word in model.wv]
    
    # If no known words, return a distinct OOV vector instead of zeros
    if not vectors:
        return np.full(vector_size, -1)  # Use a distinguishable value
    
    return np.mean(vectors, axis=0)

X_train = np.array([review_to_vector(review, w2v_model) for review in X_train_texts])
X_test = np.array([review_to_vector(review, w2v_model) for review in X_test_texts])

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Create DataLoader
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32)

# Define MIANA model
class MIANA(nn.Module):
    def __init__(self, input_size=300, hidden_size=150, dropout=0.5):
        super(MIANA, self).__init__()
        self.bi_lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)  

    def forward(self, x):
        lstm_out, _ = self.bi_lstm(x.unsqueeze(1))  
        lstm_out = self.dropout(lstm_out)  
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        weighted_sum = torch.sum(attention_weights * lstm_out, dim=1)
        output = self.fc(weighted_sum)
        return self.sigmoid(output)

# Load or train the model
MODEL_PATH = "miana_model.pth"
model = MIANA(input_size=300, hidden_size=150)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Model loaded successfully.")
else:
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Compute class weights dynamically
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    # Corrected BCELoss application
    criterion = nn.BCELoss()

    # Training loop
    for epoch in range(20):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze(1)
            loss = criterion(predictions, y_batch.squeeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)

# Evaluate the model
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        predictions = model(X_batch).squeeze(1)
        predicted_labels = (predictions > 0.5).float()  
        correct += (predicted_labels == y_batch.squeeze(1)).sum().item()
        total += y_batch.size(0)

print(f'Accuracy on test set: {correct / total * 100:.2f}%')

# Flask API for predictions
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detector')
def detector():
    return render_template('detector.html')  

@app.route('/admin_login')
def admin_login():
    return render_template('admin_login.html')  

@app.route('/admin')
def admin():
    return render_template('admin.html')  

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        review_text = data.get('review', '')
        
        if not review_text.strip() or len(word_tokenize(review_text)) < 3:
            return jsonify({'fake_review_probability': 1.0, 'is_fake': True})
        
        tokens = preprocess_text(review_text)
        vector = review_to_vector(tokens, w2v_model)
        tensor_input = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            probability = model(tensor_input).item()
        predicted_label = 1 if probability > 0.5 else 0  

        return jsonify({'fake_review_probability': probability, 'is_fake': bool(predicted_label)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run Flask server
if __name__ == '__main__':
    app.run(debug=True)
