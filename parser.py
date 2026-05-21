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
print('import done') 
    
def dataloader_train(dft):
    import pandas as pd
    import numpy as np
    try: 
        df = pd.read_csv(dft) if os.path.exists(dft) else pd.DataFrame()
        print(f"Loaded csv: {dft}") 
        return df
    except FileNotFoundError as e:
        raise e
def dataloader_test(dfT):
    import pandas as pd
    import numpy as np
    try: 
        df = pd.read_csv(dfT) if os.path.exists(dfT) else pd.DataFrame()
        print(f"Loaded csv: {dfT}") 
        return df
    except FileNotFoundError as e:
        raise e

class NeuralNetwork(nn.Module):
    def __init__(self, train_df, test_df, acc_fn, test_f_acc, train_f_acc):
        super().__init__()
        self.data_loader_train = dataloader_train(train_df)
        self.data_loader_test = dataloader_test(test_df)
        #self.acc_fn = acc_fn(k)
        
        self.linear_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        
    def forward(self, x):
        logits = self.linear_stack(x)
        softmax = nn.Softmax(dim=1)
        return softmax(logits)
    
    def print_structure(self): 
        for name, param in self.named_parameters():
            print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \\n")
net = NeuralNetwork(train_df, test_df, accuracy_function, test_f_acc, train_f_acc)

net.print_structure()
        """
        
    raw_data = tutorial() 
    cleaned_code = textwrap.dedent(raw_data).strip()
    import os 
    
    exec_context = {
        "os": os,
        "train_df": "out.csv", 
        "test_df": "out.csv", 
        "accuracy_function": accuracy_function, # Make sure this is imported/defined in your main file!
        "test_f_acc": (None, None),
        "train_f_acc": (None, None)
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

system_run()
generate_format()