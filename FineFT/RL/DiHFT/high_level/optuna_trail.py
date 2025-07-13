import os
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def objective(trial):
    trial_id = trial.number
    gpu_id = trial_id % torch.cuda.device_count()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 定义更大的超参数搜索空间
    input_size = 10
    output_size = 1
    hidden_size = trial.suggest_int('hidden_size', 16, 512)  # 更大的范围
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)  # 更大的范围
    batch_size = trial.suggest_int('batch_size', 16, 512)  # 更大的范围
    
    X = torch.randn(1000, input_size)
    y = torch.randn(1000, output_size)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = SimpleNet(input_size, hidden_size, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(10):
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    
    return val_loss / len(dataloader)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10, n_jobs=10)
print("Best hyperparameters: ", study.best_params)
