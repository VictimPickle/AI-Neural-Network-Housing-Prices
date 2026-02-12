#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Neural Network for California Housing Price Prediction
AI Course - Question 2: Neural Networks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.neural_network import MLPRegressor


df = pd.read_csv("housing.csv")
df


df.dtypes

df.isnull().sum()


(df == 0).sum()


# ======================================================================
# با توجه به اطلاعات بالا متوجه شدیم عددی نیست که مقدار نال یا 0 داشته باشد پس مشکلی نیست بجز مورد تعداد اتاق خواب ها
# ======================================================================


target_col = "median_house_value"
q1 = df[target_col].quantile(0.25)
q3 = df[target_col].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
df = df[(df[target_col] >= lower) & (df[target_col] <= upper)]


# ======================================================================
# <div dir = "rtl">
# آنهایی که خارج از چارک چهارم و چارک اول هستند را ما به عنوان داده پرت درنظر میگیریم
# </div>
# ======================================================================


X = df.drop(columns=[target_col])
y = df[target_col].to_numpy().reshape(-1, 1)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=0
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=0
)

num_cols = X_train.select_dtypes(include=[np.number]).columns
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns



num_imputer = SimpleImputer(strategy="mean")
cat_imputer = SimpleImputer(strategy="most_frequent")

X_train_num = num_imputer.fit_transform(X_train[num_cols])
X_val_num   = num_imputer.transform(X_val[num_cols])
X_test_num  = num_imputer.transform(X_test[num_cols])

if len(cat_cols) > 0:
    X_train_cat_raw = cat_imputer.fit_transform(X_train[cat_cols])
    X_val_cat_raw   = cat_imputer.transform(X_val[cat_cols])
    X_test_cat_raw  = cat_imputer.transform(X_test[cat_cols])
else:
    X_train_cat_raw = None



if X_train_cat_raw is not None:
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_train_cat = enc.fit_transform(X_train_cat_raw)
    X_val_cat   = enc.transform(X_val_cat_raw)
    X_test_cat  = enc.transform(X_test_cat_raw)
else:
    X_train_cat = np.empty((X_train_num.shape[0], 0))
    X_val_cat   = np.empty((X_val_num.shape[0], 0))
    X_test_cat  = np.empty((X_test_num.shape[0], 0))



scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_val_num_scaled   = scaler.transform(X_val_num)
X_test_num_scaled  = scaler.transform(X_test_num)



X_train_final = np.hstack([X_train_num_scaled, X_train_cat])
X_val_final   = np.hstack([X_val_num_scaled,   X_val_cat])
X_test_final  = np.hstack([X_test_num_scaled,  X_test_cat])



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader



X_train_t = torch.tensor(X_train_final, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

X_val_t = torch.tensor(X_val_final, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)

X_test_t = torch.tensor(X_test_final, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)



input_size = X_train_final.shape[1]
print("Input size (number of features):", input_size)



class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNN(input_size)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters in SimpleNN: {total_params}")



batch_size = 64
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val_t, y_val_t)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



def train_model(model, train_loader, val_loader, epochs=60, lr=0.1):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_r2": [],
        "val_r2": []
    }
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_preds_all = []
        train_targets_all = []
        train_loss_sum = 0.0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item() * X_batch.size(0)
            train_preds_all.append(outputs.detach().numpy())
            train_targets_all.append(y_batch.numpy())
        
        train_loss = train_loss_sum / len(train_loader.dataset)
        train_preds_all = np.vstack(train_preds_all)
        train_targets_all = np.vstack(train_targets_all)
        train_r2 = r2_score_custom(train_targets_all, train_preds_all)
        
        # Validation
        model.eval()
        val_preds_all = []
        val_targets_all = []
        val_loss_sum = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss_sum += loss.item() * X_batch.size(0)
                val_preds_all.append(outputs.numpy())
                val_targets_all.append(y_batch.numpy())
        
        val_loss = val_loss_sum / len(val_loader.dataset)
        val_preds_all = np.vstack(val_preds_all)
        val_targets_all = np.vstack(val_targets_all)
        val_r2 = r2_score_custom(val_targets_all, val_preds_all)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_r2"].append(train_r2)
        history["val_r2"].append(val_r2)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}")
    
    return history



def r2_score_custom(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def mae_custom(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mse_custom(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rmse_custom(y_true, y_pred):
    return np.sqrt(mse_custom(y_true, y_pred))



history = train_model(model, train_loader, val_loader, epochs=60, lr=0.1)



model.eval()
with torch.no_grad():
    train_preds = model(X_train_t).numpy()
    val_preds = model(X_val_t).numpy()
    test_preds = model(X_test_t).numpy()

print("=== Simple Network Evaluation ===")
print("\nTrain Set:")
print(f"R²:   {r2_score_custom(y_train, train_preds):.4f}")
print(f"MAE:  {mae_custom(y_train, train_preds):.4f}")
print(f"MSE:  {mse_custom(y_train, train_preds):.4f}")
print(f"RMSE: {rmse_custom(y_train, train_preds):.4f}")

print("\nValidation Set:")
print(f"R²:   {r2_score_custom(y_val, val_preds):.4f}")
print(f"MAE:  {mae_custom(y_val, val_preds):.4f}")
print(f"MSE:  {mse_custom(y_val, val_preds):.4f}")
print(f"RMSE: {rmse_custom(y_val, val_preds):.4f}")

print("\nTest Set:")
print(f"R²:   {r2_score_custom(y_test, test_preds):.4f}")
print(f"MAE:  {mae_custom(y_test, test_preds):.4f}")
print(f"MSE:  {mse_custom(y_test, test_preds):.4f}")
print(f"RMSE: {rmse_custom(y_test, test_preds):.4f}")



plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Loss over Epochs - Simple Network")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history["train_r2"], label="Train R²")
plt.plot(history["val_r2"], label="Val R²")
plt.xlabel("Epoch")
plt.ylabel("R²")
plt.title("R² over Epochs - Simple Network")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



class ComplexNN(nn.Module):
    def __init__(self, input_size):
        super(ComplexNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        
        self.fc4 = nn.Linear(16, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.relu3(x)
        
        x = self.fc4(x)
        return x

complex_model = ComplexNN(input_size)

total_params_complex = sum(p.numel() for p in complex_model.parameters())
print(f"Total parameters in ComplexNN: {total_params_complex}")



history_complex = train_model(complex_model, train_loader, val_loader, epochs=60, lr=0.01)



complex_model.eval()
with torch.no_grad():
    train_preds_complex = complex_model(X_train_t).numpy()
    val_preds_complex = complex_model(X_val_t).numpy()
    test_preds_complex = complex_model(X_test_t).numpy()

print("=== Complex Network Evaluation ===")
print("\nTrain Set:")
print(f"R²:   {r2_score_custom(y_train, train_preds_complex):.4f}")
print(f"MAE:  {mae_custom(y_train, train_preds_complex):.4f}")
print(f"MSE:  {mse_custom(y_train, train_preds_complex):.4f}")
print(f"RMSE: {rmse_custom(y_train, train_preds_complex):.4f}")

print("\nValidation Set:")
print(f"R²:   {r2_score_custom(y_val, val_preds_complex):.4f}")
print(f"MAE:  {mae_custom(y_val, val_preds_complex):.4f}")
print(f"MSE:  {mse_custom(y_val, val_preds_complex):.4f}")
print(f"RMSE: {rmse_custom(y_val, val_preds_complex):.4f}")

print("\nTest Set:")
print(f"R²:   {r2_score_custom(y_test, test_preds_complex):.4f}")
print(f"MAE:  {mae_custom(y_test, test_preds_complex):.4f}")
print(f"MSE:  {mse_custom(y_test, test_preds_complex):.4f}")
print(f"RMSE: {rmse_custom(y_test, test_preds_complex):.4f}")



plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_complex["train_loss"], label="Train Loss")
plt.plot(history_complex["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Loss over Epochs - Complex Network")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_complex["train_r2"], label="Train R²")
plt.plot(history_complex["val_r2"], label="Val R²")
plt.xlabel("Epoch")
plt.ylabel("R²")
plt.title("R² over Epochs - Complex Network")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 8))
plt.scatter(y_test, test_preds_complex, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs Actual - Complex Network (Test Set)")
plt.grid(True)
plt.tight_layout()
plt.show()



print("\n=== Model Comparison ===")
print("\nSimple Network:")
print(f"Parameters: {total_params}")
print(f"Test R²: {r2_score_custom(y_test, test_preds):.4f}")
print(f"Test MAE: {mae_custom(y_test, test_preds):.4f}")
print(f"Test RMSE: {rmse_custom(y_test, test_preds):.4f}")

print("\nComplex Network:")
print(f"Parameters: {total_params_complex}")
print(f"Test R²: {r2_score_custom(y_test, test_preds_complex):.4f}")
print(f"Test MAE: {mae_custom(y_test, test_preds_complex):.4f}")
print(f"Test RMSE: {rmse_custom(y_test, test_preds_complex):.4f}")

print("\nConclusion:")
if r2_score_custom(y_test, test_preds_complex) > r2_score_custom(y_test, test_preds):
    print("The complex network performs better with higher R² and lower errors.")
    print("Additional layers and dropout help capture more complex patterns.")
else:
    print("The simple network is sufficient for this task.")

