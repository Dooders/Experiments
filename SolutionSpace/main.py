import random
from collections import deque
from typing import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

SEED = 42
RECORD_COUNT = 1000
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

transformation_map = OrderedDict(
    {
        "glucose": -1.0,
        "atp": 2.0,
        "adp": 2.0,
        "nad+": -2.0,
        "nadh": 2.0,
        "pi": -2.0,
        "pyruvate": 2.0,
        "h2o": 2.0,
    }
)

transformation_template = [transformation_map[key] for key in transformation_map.keys()]


# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        torch.manual_seed(SEED)
        self.forward_count = 0  # Initialize counter
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, input_size),
        )

    def forward(self, x):
        self.forward_count += x.shape[0]  # Increment by batch size
        return self.layers(x)

    def get_forward_count(self):
        return self.forward_count


# Custom dataset
class VectorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Define the RL agent with experience replay
class PredictorAgent:
    def __init__(self, state_size, memory_size=32, batch_size=32):
        self.state_size = state_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

        self.model = SimpleNN(state_size)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()

    def remember(self, state, target):
        self.memory.append((state, target))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, targets = zip(*batch)

        states = torch.FloatTensor(states)
        targets = torch.FloatTensor(targets)

        self.optimizer.zero_grad()
        predictions = self.model(states)
        loss = self.criterion(predictions, targets)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            return self.model(state).squeeze(0).numpy()


def generate_random_input(length: int):
    return [float(random.randint(0, 100)) for _ in range(length)]


# Function to get model weights as numpy arrays
def get_model_weights(model):
    return {
        name: param.data.clone().numpy() for name, param in model.named_parameters()
    }


# Generate training data (same as before)
X = np.array(
    [generate_random_input(len(transformation_template)) for _ in range(RECORD_COUNT)]
)
y = np.array([x + transformation_template for x in X])

# Create dataset and split into train/test
dataset = VectorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize model, loss function, and optimizer
model = SimpleNN(len(transformation_template))
initial_trained_weights = get_model_weights(model)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
num_epochs = 300
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            train_loss = sum(
                criterion(model(batch_X), batch_y) for batch_X, batch_y in train_loader
            ) / len(train_loader)
            test_loss = sum(
                criterion(model(batch_X), batch_y) for batch_X, batch_y in test_loader
            ) / len(test_loader)
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
            )

# Test prediction on our input vector
model.eval()
with torch.no_grad():
    input_vector = generate_random_input(len(transformation_template))
    input_tensor = torch.FloatTensor(input_vector).unsqueeze(0)
    prediction = model(input_tensor)
    print("\nPrediction for input vector:")
    print(prediction[0].numpy())
    print("\nActual result should be:")
    print(
        [input_vector[i] + transformation_template[i] for i in range(len(input_vector))]
    )

final_trained_weights = get_model_weights(model)


# Training the RL agent
def train_rl_agent(pytorch_model, rl_agent, train_loader, all_losses, num_epochs=300):
    for epoch in range(num_epochs):
        total_loss = 0
        for (
            batch_X,
            batch_y,
        ) in (
            train_loader
        ):  # We don't need batch_y since we're using pytorch_model's output
            # Store and train on each sample in the batch
            batch_loss = 0
            for state, target in zip(batch_X.numpy(), batch_y.numpy()):
                rl_agent.remember(state, target)
                loss = rl_agent.train()
                if loss is not None:
                    batch_loss += loss
            all_losses.append(batch_loss)
            total_loss += batch_loss

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader.dataset)
            print(f"Episode [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")


all_losses = []

# Update the training call
rl_agent = PredictorAgent(len(transformation_template))
initial_rl_weights = get_model_weights(rl_agent.model)
train_rl_agent(model, rl_agent, train_loader, all_losses)
final_rl_weights = get_model_weights(rl_agent.model)

# Test the RL agent on the same input vector
print("\nRL Agent Prediction:")
rl_prediction = rl_agent.predict(input_vector)
print(rl_prediction)

print("\nPyTorch Model Prediction:")
with torch.no_grad():
    pytorch_prediction = (
        model(torch.FloatTensor(input_vector).unsqueeze(0)).squeeze(0).numpy()
    )
print(pytorch_prediction)


# Function to get model weights as numpy arrays
def get_model_weights(model):
    return {name: param.data.numpy() for name, param in model.named_parameters()}


# Get weights from both models
pytorch_weights = get_model_weights(model)
rl_weights = get_model_weights(rl_agent.model)

# Compare weights
print("\nWeight Comparison:")
for name in pytorch_weights:
    print(f"\n{name}:")
    print(f"PyTorch Model shape: {pytorch_weights[name].shape}")
    print(f"RL Agent shape: {rl_weights[name].shape}")

    # Calculate difference statistics
    weight_diff = np.abs(pytorch_weights[name] - rl_weights[name])
    print(f"Mean absolute difference: {np.mean(weight_diff):.6f}")
    print(f"Max absolute difference: {np.max(weight_diff):.6f}")

# After PyTorch model training
print(f"\nPyTorch model forward passes: {model.get_forward_count()}")

# After RL agent training
print(f"RL agent forward passes: {rl_agent.model.get_forward_count()}")

def weights_to_pca(weights_dict):
    """Flatten and transform weights to 2D using PCA"""
    # Flatten all weights into a single vector
    flattened = np.concatenate([w.flatten() for w in weights_dict.values()])
    
    # Create multiple samples by adding small random noise
    n_samples = 100
    noise_scale = 0.01
    samples = np.array([
        flattened + np.random.normal(0, noise_scale, size=flattened.shape) 
        for _ in range(n_samples)
    ])
    
    # Transform to 2D
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(samples)
    
    # Return the transformation of the original weights (first sample)
    return transformed[0]


# Transform weights to 2D for visualization
initial_pytorch_pca = weights_to_pca(initial_trained_weights)
final_pytorch_pca = weights_to_pca(final_trained_weights)
initial_rl_pca = weights_to_pca(initial_rl_weights)
final_rl_pca = weights_to_pca(final_rl_weights)

print("\nPCA Transformed Weights (2D):")
print(f"Initial PyTorch weights: {initial_pytorch_pca}")
print(f"Final PyTorch weights: {final_pytorch_pca}")
print(f"Initial RL weights: {initial_rl_pca}")
print(f"Final RL weights: {final_rl_pca}")

# Create scatter plot
plt.figure(figsize=(10, 8))

# Plot PyTorch model points
plt.scatter(initial_pytorch_pca[0], initial_pytorch_pca[1], color='blue', marker='o', s=100, label='PyTorch Initial')
plt.scatter(final_pytorch_pca[0], final_pytorch_pca[1], color='blue', marker='^', s=100, label='PyTorch Final')

# Plot RL model points
plt.scatter(initial_rl_pca[0], initial_rl_pca[1], color='red', marker='o', s=100, label='RL Initial')
plt.scatter(final_rl_pca[0], final_rl_pca[1], color='red', marker='^', s=100, label='RL Final')

# Draw arrows to show the training progression
plt.arrow(initial_pytorch_pca[0], initial_pytorch_pca[1], 
          final_pytorch_pca[0] - initial_pytorch_pca[0], 
          final_pytorch_pca[1] - initial_pytorch_pca[1],
          color='blue', alpha=0.3, head_width=0.1)
plt.arrow(initial_rl_pca[0], initial_rl_pca[1], 
          final_rl_pca[0] - initial_rl_pca[0], 
          final_rl_pca[1] - initial_rl_pca[1],
          color='red', alpha=0.3, head_width=0.1)

plt.title('Model Weight Space Trajectories (PCA)')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.grid(True)
plt.show()

# Print parameters
print(f"Initial PyTorch weights: {initial_trained_weights}")
print('***********************************')
print(f"Initial RL weights: {initial_rl_weights}")


