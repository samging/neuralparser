import torch
from torch import nn
import pandas as pd
import numpy as np
import os

print('import done') 
    
def dataloader_train(dft):
    import pandas as pd
    try: 
        df = pd.read_csv(dft) if os.path.exists(dft) else pd.DataFrame()
        print(f"Loaded train csv: {dft}")
        if df.empty:
            raise ValueError("Can't use empty dataframe")
        return df
    except Exception as e:
        raise e

def dataloader_test(dfT):
    import pandas as pd
    try: 
        df = pd.read_csv(dfT) if os.path.exists(dfT) else pd.DataFrame()
        print(f"Loaded test csv: {dfT}") 
        if df.empty:
            raise ValueError("Can't use empty dataframe")
        return df
    except Exception as e:
        raise e

class NeuralNetwork(nn.Module):
    def __init__(self, train_df, test_df, layer_sizes, net_id):
        super().__init__()
        self.data_loader_train = dataloader_train(train_df)
        self.data_loader_test = dataloader_test(test_df)
        self.net_id = net_id
        
        self.layers = []
        input_dim = 28 * 28
        
        for hidden_dim in layer_sizes:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            input_dim = hidden_dim
            
        self.layers.append(nn.Linear(input_dim, 10))
        
        self.linear_stack = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.linear_stack(x)
            
    def train_and_evaluate(self, epochs=1000):
        print(f"\\n--- Training Network Architecture #{self.net_id} ---")
        if self.data_loader_train.empty:
            print("Data empty. Skipping execution loop.")
            return
            
        labels = self.data_loader_train.iloc[:, 0].values.astype(np.int64)
        features = self.data_loader_train.iloc[:, 1:785].values.astype(np.float32) / 255.0
        
        X = torch.tensor(features)
        y = torch.tensor(labels)
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        history = []
        
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = self.forward(X[:4000]) 
            loss = loss_fn(pred, y[:4000])
            loss.backward()
            optimizer.step()
            
            history.append(loss.item())
            if (epoch + 1) % 10 == 0:
                print(f"Net {self.net_id} | Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
        
        if not self.data_loader_test.empty:
            test_features = self.data_loader_test.iloc[:, 1:785].values.astype(np.float32) / 255.0
            input_test = torch.tensor(test_features)
            
            self.eval()
            with torch.no_grad():
                logits = self.forward(input_test)
                probabilities = nn.Softmax(dim=1)(logits)
                predictions = torch.argmax(probabilities, dim=1)
            
            output_preds = pd.DataFrame({
                "predicted_digit": predictions.cpu().numpy(),
                "confidence_score": torch.max(probabilities, dim=1).values.cpu().numpy(),
                "neural_structure": str(self.linear_stack)
            }) 
            output_preds.to_csv(f"trained_results_net_{self.net_id}.csv", index=False)
            
            loss_df = pd.DataFrame({"epoch_loss": history})
            loss_df.to_csv(f"loss_history_net_{self.net_id}.csv", index=False)
            print(f"Saved configuration data blocks for Network #{self.net_id}!")

# Read dynamic configs injected via context mapping loops
net = NeuralNetwork(train_df, test_df, layer_configuration, network_index)
net.train_and_evaluate(epochs=100)