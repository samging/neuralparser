import pandas as pd
import numpy as np

def prep_env(): 
    import subprocess
    import os
    
    venv_dir = os.path.join(os.getcwd(), 'pytorch')

    if not os.path.exists(venv_dir):
        subprocess.run(['python3', '-m', 'venv', venv_dir])
        print("!!\t Created new virtual environment")
    
    venv_pip = os.path.join(venv_dir, 'bin', 'pip3')
    
    result = subprocess.run([venv_pip, 'list'], capture_output=True, text=True)
    
    if 'torch' not in result.stdout.lower():
        print("PyTorch not found. Installing...")
        subprocess.run([venv_pip, 'install', 'torch'])
    else:
        print("PyTorch is already installed!")

    subprocess.run(['ls', '-la'])
    return venv_dir  #name for activation



def accuracy_function(prediction: tuple, true_label: tuple, *args, **kwargs):
    """
    TUPLE: (str, str) 
    
    expects: csv columns to be passed as parameters
    positional parameters: 
      prediction: (column_name, file_path)
      true_label: (column_name, file_path)
    """ 
    try:
        true_label_df = pd.read_csv(true_label[1], usecols=[true_label[0]])[true_label[0]]
        pred_label_df = pd.read_csv(prediction[1], usecols=[prediction[0]])[prediction[0]]
    except FileNotFoundError as e:
        raise e
    
    structure_data = {}
    
    if args:
        match args[0]: 
            case "write_nn":
                structure_index = int(args[1] + 1) if len(args) > 1 else 1
                structure_data[f"structure_{structure_index}"] = [(k, v) for k, v in kwargs.items()]
            case _:
                pass

    max_true = max(true_label_df)
    max_pred = max(pred_label_df)
    math_results = [np.abs(T - p) / max_true + max_pred for T in true_label_df for p in pred_label_df]
    
    print(f"Calculated math results size: {len(math_results)}")
    
    final_df = pd.DataFrame({
        "calculated_error": math_results
    })
    
    final_df.to_csv("acc_f_struct.csv", index=False)
    
    return final_df

def install_dep(dep:str = ""): 
    import subprocess 
    import os
    import sys
    
    strip_prefx = sys.prefix.strip('/')
    venv_name = strip_prefx.split('/')[-1]
    venv_filepath = os.path.join(os.getcwd(), 'pytorch')
    
    if venv_name == 'pytorch':
        print("you can insert your download pip package requirements then it will check and remember")
        packages = subprocess.run(['pip3', 'list'], capture_output=True, text=True)
        print(packages)
        if str(dep) not in packages:
            try:  
                result = subprocess.run(
                    ['pip3', 'install', str(dep)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"Pip crashed due to: {e.stderr} code: {e.returncode}")
                raise RuntimeError(f"Installation failed") from e
            

def generate_format(n: int = 10, *args):
    import textwrap 
    def tutorial() -> str: 
        print("tutorial called") 
        return """import torch
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
    
    def save_net(self): 
        print(f"Net_ID: {self.net_id} (Saving)")
        
        neural_metrics = pd.DataFrame({
            "network_id": [f"net_{self.net_id}"],
            "architecture": [str(self.linear_stack)]
        })
        
        filename = "NeuralNetwork.csv"
        file_exists = os.path.exists(filename)
        neural_metrics.to_csv(filename, mode='a', index=False, header=not file_exists) 
        
    def forward(self, x):
        return self.linear_stack(x)
            
    def train_and_evaluate(self, epochs=1000):
        print(f"\\n--- Training Network Architecture #{self.net_id} ---")
        if self.data_loader_train.empty:
            print("Data empty. Skipping execution loop.")
            return
            
        self.save_net()
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
                "confidence_score": torch.max(probabilities, dim=1).values.cpu().numpy()
            }) 
            output_preds.to_csv(f"trained_results_net_{self.net_id}.csv", index=False)
            
            loss_df = pd.DataFrame({"epoch_loss": history})
            loss_df.to_csv(f"loss_history_net_{self.net_id}.csv", index=False)
            print(f"Saved configuration data blocks for Network #{self.net_id}!")
        
net = NeuralNetwork(train_df, test_df, layer_configuration, network_index)
net.train_and_evaluate(epochs=10)
"""
    import os 
    raw_data = tutorial() 
    cleaned_code = textwrap.dedent(raw_data).strip()
    
    architectures = [
        [512], [256], [128],                   # Single hidden layer variants
        [512, 256], [256, 128], [512, 128],     # Dual hidden layer variants
        [512, 256, 128], [256, 256, 256],       # Deep structural blocks
        [128, 64], [256, 64], [512, 64]         # Bottleneck structure configurations
    ]
    
    while len(architectures) < 5:
        architectures.append([512 - (len(architectures) * 10), 128])

    # TRIGGER SYSTEM LOOP RUNS OVER ALL 30 ARCHITECTURES
    for idx, config in enumerate(architectures):
        exec_context = {
            "os": os,
            "train_df": "fashion-mnist_train.csv", 
            "test_df": "fashion-mnist_test.csv", 
            "accuracy_function": accuracy_function,
            "test_f_acc": (None, None),
            "train_f_acc": (None, None),
            "layer_configuration": config,        # Injects current layout size array
            "network_index": idx + 1              # Tracks numerical name of script
        } 
        exec(cleaned_code, exec_context)


def system_run(): 
    import os 
    import subprocess
     
    custom_env = os.environ.copy()
    
    if "PARSER_LOOP_BREAK" not in os.environ: 
        custom_env["PARSER_LOOP_BREAK"] = 'false'
    else:
        custom_env["PARSER_LOOP_BREAK"] = os.environ["PARSER_LOOP_BREAK"]
    
    loop_status = custom_env.get("PARSER_LOOP_BREAK") 
    
    if loop_status == "true":
        print("Exiting the loop break") 
        return

    
    if loop_status == "false":
        print("\nstarting the loop") 
        venv_path = prep_env()
        parent_path = os.path.dirname(venv_path)
        os.chdir(parent_path)
        source_dest = os.path.join(venv_path, 'bin', 'activate')
        command = f"source {source_dest} && python3 parser.py"
        custom_env["PARSER_LOOP_BREAK"] = 'true'
        
        subprocess.run([command], shell=True, executable='/bin/bash',env=custom_env)
        exit(0)


#prep_env()
system_run()
generate_format()