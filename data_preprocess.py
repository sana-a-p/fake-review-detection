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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
data = pd.read_csv('yelp.csv')

# Preprocess text
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(str(text).lower())  # Ensure text is string
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

data['cleaned_reviews'] = data['text'].apply(preprocess_text)

# Train Word2Vec model
w2v_model = Word2Vec(sentences=data['cleaned_reviews'], vector_size=300, window=5, min_count=2, workers=4)

# Convert reviews into Word2Vec embeddings
def review_to_vector(review, model, vector_size=300):
    vectors = [model.wv[word] for word in review if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

X = np.array([review_to_vector(review, w2v_model) for review in data['cleaned_reviews']])
y = data['validity'].values  # Labels from 'useful' column

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)  # Ensure correct shape for labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define MIANA model without embedding layer
class MIANA(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MIANA, self).__init__()
        self.bi_lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_size * 2, 1)  # Attention layer
        self.fc = nn.Linear(hidden_size * 2, 1)  # Fully connected layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is expected to be (batch_size, seq_len, input_size)
        lstm_out, _ = self.bi_lstm(x)

        # Attention weights: (batch_size, seq_len, 1)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)

        # Weighted sum of LSTM outputs: (batch_size, hidden_size * 2)
        weighted_sum = torch.sum(attention_weights * lstm_out, dim=1)

        output = self.fc(weighted_sum)  # Now it should be (batch_size, 1)
        return self.sigmoid(output)  # Apply sigmoid for binary classification

# Initialize model
model = MIANA(input_size=300, hidden_size=150)

# Initialize optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training loop
total_loss = 0  # Initialize total_loss before starting the loop
for X_batch, y_batch in train_loader:
    optimizer.zero_grad()

    # Ensure the input is in the shape (batch_size, seq_len, input_size)
    X_batch = X_batch.unsqueeze(1)  # Add sequence dimension (seq_len = 1)
    print(f"Shape of X_batch: {X_batch.shape}")  # Debugging: Check shape of X_batch

    predictions = model(X_batch).squeeze(1)  # Ensure predictions match y_batch shape
    loss = criterion(predictions, y_batch)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

print(f"Total loss after training: {total_loss}")

# Evaluate the model on the test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        # Ensure the input is in the shape (batch_size, seq_len, input_size)
        X_batch = X_batch.unsqueeze(1)  # Add sequence dimension (seq_len = 1)

        predictions = model(X_batch).squeeze(1)
        predicted_labels = (predictions > 0.5).float()  # Threshold at 0.5 for binary classification
        correct += (predicted_labels == y_batch).sum().item()
        total += y_batch.size(0)

accuracy = correct / total
print(f'Accuracy on test set: {accuracy * 100:.2f}%')