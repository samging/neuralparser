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
    for _ in range(0,n):
       exec('print(f"nn.{str(args[0])}({int(args[1])}, {int(args[2])})")')

#generate_format(10,'Linear', 10,100)
prep_env()
install_dep('pytorch')
install_dep('matplotlib')
install_dep() 