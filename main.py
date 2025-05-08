import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --- Device configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Preprocessing ---

def preprocess_qos_data(data: pd.DataFrame):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

def preprocess_chat_data(chats, tokenizer, max_length=128):
    return tokenizer(
        chats,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

# --- QoS Classifier ---

class QoSClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super(QoSClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

# --- BERT Model for Sentiment Analysis ---

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sentiment_model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2
).to(device)
sentiment_model.eval()

def sentiment_analysis(chats):
    inputs = preprocess_chat_data(chats, tokenizer)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    return predictions.cpu()

# --- Dynamic QoS Management ---

def adjust_qos(qos_pred, sentiment_pred):
    if qos_pred == 1 or sentiment_pred == 1:
        print("Increasing bandwidth and reducing latency.")
    else:
        print("Stable QoS parameters.")

# --- Main Loop ---

def main_loop(qos_df: pd.DataFrame, chat_data: list[str]):
    scaled_qos, _ = preprocess_qos_data(qos_df)
    model = QoSClassifier(input_dim=scaled_qos.shape[1]).to(device)
    model.eval()

    for idx, (metrics, chat) in enumerate(zip(scaled_qos, chat_data)):
        metrics_tensor = torch.tensor(metrics, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            qos_pred = model(metrics_tensor)
        sentiment_pred = sentiment_analysis([chat])

        adjust_qos(qos_pred.argmax().item(), sentiment_pred.item())
        print(f"Iteration {idx+1}: QoS Prediction: {qos_pred}, Chat Sentiment: {sentiment_pred}")

# --- Sample Runs ---

# 1. Regular stream
qos_data = pd.DataFrame({
    'latency': [20, 30, 25, 40],
    'bitrate': [5000, 4500, 5200, 4800],
    'packet_loss': [0.1, 0.05, 0.07, 0.02]
})
chat_data = [
    "This stream is amazing!",
    "Why is it lagging so much?",
    "The quality is really good today.",
    "Buffering again... not great."
]
main_loop(qos_data, chat_data)

# 2. High latency and packet loss
qos_data = pd.DataFrame({
    'latency': [100, 150, 120, 130],
    'bitrate': [2000, 1800, 2100, 1900],
    'packet_loss': [0.15, 0.20, 0.18, 0.17]
})
chat_data = [
    "The stream is lagging so much!",
    "Can't watch, it's buffering every minute.",
    "What's going on with the quality today?",
    "This is unwatchable!"
]
main_loop(qos_data, chat_data)

# 3. Smooth streaming
qos_data = pd.DataFrame({
    'latency': [10, 12, 8, 15],
    'bitrate': [6000, 6200, 5900, 6100],
    'packet_loss': [0.01, 0.02, 0.00, 0.01]
})
chat_data = [
    "This stream is crystal clear!",
    "Loving the quality today!",
    "Perfect stream, no lag at all.",
    "Best quality I've seen so far!"
]
main_loop(qos_data, chat_data)

# 4. Mixed quality
qos_data = pd.DataFrame({
    'latency': [25, 45, 20, 60],
    'bitrate': [5000, 4800, 5300, 4700],
    'packet_loss': [0.05, 0.10, 0.03, 0.12]
})
chat_data = [
    "It's fine most of the time, but sometimes it lags.",
    "A bit choppy but still watchable.",
    "Smooth at first, but now it's buffering a lot.",
    "Why is it stuttering so much now?"
]
main_loop(qos_data, chat_data)

# 5. Idle stream
qos_data = pd.DataFrame({
    'latency': [5, 5, 5, 5],
    'bitrate': [1000, 1000, 1000, 1000],
    'packet_loss': [0.00, 0.00, 0.00, 0.00]
})
chat_data = [
    "Stream is okay.",
    "Nothing much happening right now.",
    "Just waiting for the game to start.",
    "Chat is kinda quiet today."
]
main_loop(qos_data, chat_data)

# 6. High quality (4K)
qos_data = pd.DataFrame({
    'latency': [10, 8, 12, 9],
    'bitrate': [8000, 8200, 7900, 8100],
    'packet_loss': [0.02, 0.01, 0.01, 0.00]
})
chat_data = [
    "The stream is so smooth!",
    "I love the 4K quality, it's amazing.",
    "Great job with the stream, perfect clarity.",
    "This is e-sports quality for sure!"
]
main_loop(qos_data, chat_data)
