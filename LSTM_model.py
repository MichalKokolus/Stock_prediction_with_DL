# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:07:05 2024

@author: kokol
"""

import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch
import torch.optim as optim
import torch.cuda as cuda
from sklearn.preprocessing import MinMaxScaler
import os
from os.path import abspath, join

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = pd.read_csv("C:\\Users\\kokol\\Documents\\Personal\\03_My_projects\\01_LLMs\\Stock_prediction_with_DL\\MSFT_data.csv", index_col="Date", parse_dates=True)
close_prices = data["Close"]


scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(close_prices.values.reshape(-1, 1))

# Define sequence length (number of past closing prices to consider)
sequence_length = 5

# Create sequences of closing prices
sequences = []
for i in range(len(normalized_data) - sequence_length):
    sequence = normalized_data[i:i + sequence_length]
    sequences.append(sequence)

# Convert sequences to tensors
sequences_tensor = torch.tensor(sequences)

# Split data into training and testing sets (optional)
# You can use techniques like train-test split or time series splitting
# for evaluation purposes
train_size = int(len(sequences_tensor) * 0.8)  # 80% for training
train_data = sequences_tensor[:train_size]
test_data = sequences_tensor[train_size:]

# Create training and testing dataloaders (optional, but improves efficiency)
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)  # Adjust batch size as needed
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # Initialize hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # Initialize cell state
        out, (hn, cn) = self.lstm(x, (h0, c0))  # Pass input through LSTM
        out = out[:, -1, :]  # Get the last output from the sequence
        out = self.fc(out)  # Pass output through fully connected layer
        return out

learning_rate = 0.001  # Adjust as needed
epochs = 100  # Adjust as needed

# Initialize model and optimizer
model = LSTMModel(input_size=1, hidden_size=64, num_layers=1, output_size=1)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()  # Mean Squared Error loss function

# Training loop
for epoch in range(epochs):
    for i, (data) in enumerate(train_dataloader):
        inputs, labels = data.squeeze(1).float(), data[:, -1].float()  # Separate inputs and labels

        # Clear gradients before each pass
        optimizer.zero_grad()

        # Forward pass: Get predictions
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass: Propagate gradients
        loss.backward()

        # Update model weights
        optimizer.step()

        # Optionally print training progress
        if i % 100 == 0:  # Print every 100 batches
            print(f"Epoch: [{epoch + 1}/{epochs}], Batch: [{i}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

# Optional: Evaluate model performance on test data (if applicable)
with torch.no_grad():
    total_loss = 0
    num_batches = 0
    for i, (data) in enumerate(test_dataloader):
        inputs, labels = data.squeeze(1).float(), data[:, -1].float()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        num_batches += 1

    average_loss = total_loss / num_batches
    print(f"Average Test Loss: {average_loss:.4f}")


