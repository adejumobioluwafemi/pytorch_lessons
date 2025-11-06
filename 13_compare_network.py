from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch 
import torch.nn as nn
import torch.optim as optim

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def create_network_architecture(input_size, num_classes):
    """
    create different network architectures for comparison
    """

    architectures = {
        "Shallow Network": nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        ),
        "Deep Network": nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        ),
        'Network with Dropout': nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        ),
        'Network with BatchNorm': nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    }
    return architectures

def compare_architectures(X_train, y_train, X_test, y_test, input_size, num_classes, num_epochs=10):
    
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    architectures = create_network_architecture(input_size, num_classes)

    results = {}

    for name, model in architectures.items():
        print(f"\n--- Training {name} ---")
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            _, train_pred = torch.max(outputs, 1)
            train_acc = (train_pred == y_train).float().mean()
            if (epoch+1)%10==0:
                print(f"epoch {epoch+1}/{num_epochs}, loss = {loss.item():.4f}")

        # evaluate
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, test_pred = torch.max(test_outputs, 1)
            test_acc = (test_pred == y_test).float().mean()

        results[name] = {
            'train_accuracy': train_acc.item(),
            'test_accuracy': test_acc.item()
        }
    return results

X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Compare architectures
input_size = X_train.shape[1]
num_classes = len(torch.unique(y_train_tensor))

results = compare_architectures(
    X_train_tensor, y_train_tensor, 
    X_test_tensor, y_test_tensor,
    input_size, num_classes
)

print(f"\nARCHITECTURE COMPARISON RESULTS:")
for name, metrics in results.items():
    print(f"{name:25} - Train: {metrics['train_accuracy']:.4f}, Test: {metrics['test_accuracy']:.4f}")
