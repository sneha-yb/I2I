#  Step 1: Import Libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Step 2: Load Data
df = pd.read_csv("/content/fruit_features.csv")

# Step 3: Feature Selection & Target
features = ['std_power', '2nd_Peak_Mag', 'std_entropy', '3rd_Peak_Mag', 'Mean_Magnitude']
# features = [ 'std_power',  'std_entropy']
X = df[features].values
brix = df['Brix_Value'].values

# Step 4: Define Ripeness Categories
def classify_brix(brix):
    if brix < 17.5:
        return 0  # Unripe
    # elif brix < 21.5:
    #     return 1  # Ripe
    else:
        return 1  # Overripe

y = np.array([classify_brix(b) for b in brix])

#  Step 5: Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

#  Step 7: Define MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.model(x)

#  Step 8: Train the Classifier
model = MLPClassifier(input_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

#  Step 9: Evaluate the Model
model.eval()
with torch.no_grad():
    logits = model(X_test_tensor)
    predictions = torch.argmax(logits, dim=1).numpy()

#  Accuracy
acc = accuracy_score(y_test, predictions)
print(f"\n✅ Classification Accuracy: {acc:.2f}")

#  Confusion Matrix
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Unripe", "Ripe"])
disp.plot(cmap='Blues')
plt.title("Ripeness Classification Confusion Matrix")
plt.grid(False)
plt.show()

demo_df = pd.read_csv("/content/demo_test.csv")
demo_features = demo_df[features].values  # Ensure column names match original feature list

# Scale using the previously fitted scaler
demo_scaled = scaler.transform(demo_features)
demo_tensor = torch.tensor(demo_scaled, dtype=torch.float32)

# Run through the trained model
model.eval()
with torch.no_grad():
    demo_output = model(demo_tensor)
    demo_predictions = torch.argmax(demo_output, dim=1).numpy()

# Display predictions
class_names = {0: "Unripe", 1: "Ripe"}
for i, pred in enumerate(demo_predictions):
    print(f"🍌 Sample {i+1} Prediction: {class_names[pred]}")
