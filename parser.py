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
            

def generate_format(n:int = 10, *args):
    import textwrap 
    def tutorial() -> str: 
        print("tutorial called") 
        return """
        import torch
        from torch import nn
        print('import done') 
        
        class NeuralNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_stack = nn.Sequential(
                    nn.Linear(28*28, 512),
                    nn.ReLU(),
                    nn.Linear(512, 10),
                )
            def say_hi(self): 
                print('hi from neural network instance!')
        
        # Instantiate and run it
        net = NeuralNetwork()
        net.say_hi()
        """
    raw_data = tutorial() 
    cleaned_code = textwrap.dedent(raw_data).strip()
    exec(cleaned_code,{})
    


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
#print(venv_path)

#subprocess.run(venv_path, 'parser.py')
#generate_format()

#install_dep('pytorch')
#install_dep('matplotlib')
#install_dep() 