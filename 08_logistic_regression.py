# pipeline
# 0. prepare data
# 1. Design model (input size, output size, forward pass)
# 2. Construct loss and optimizer
# 3. Training loop
#   - Forward pass: compute prediction
#   - Backward pass: gradients
#   - Update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# data prep
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape
print(f'X.shape ==> {X.shape}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32)).view(y_train.shape[0], 1)
y_test = torch.from_numpy(y_test.astype(np.float32)).view(y_test.shape[0], 1)

# model
# f = wx + b, then sigmoid

class LogisticRegression(nn.Module):

    def __init__(self, n_features, num_classes):
        super(LogisticRegression,self).__init__()

        self.linear = nn.Linear(n_features, num_classes)

    def forward(self, x):
        logit = self.linear(x)
        pred = torch.sigmoid(logit)
        return pred

num_classes = 1
model = LogisticRegression(n_features=n_features, num_classes=num_classes)
lr = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

def training(model, X_train, y_train, num_epochs=100):

    for epoch in range(num_epochs):
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (epoch+1)%10 == 0:
            print(f'epoch: {epoch+1}, loss= {loss.item():.4f}')
    return model

model = training(model, X_train, y_train, num_epochs=100)

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum()/float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')