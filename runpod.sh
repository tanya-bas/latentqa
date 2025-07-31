#  !/bin/bash  
#  # Exit on any error 
 
#  set -e  
 
#  START_TIME=$SECONDS  
 
#  echo ">>> Installing uv..." 
 
#  curl -LsSf https://astral.sh/uv/install.sh | sh export PATH="$HOME/.local/bin:$PATH" uv --version # Ensure the script is run from the correct directory 
#  echo ">>> uv installed successfully!" echo ""  echo ">>> Building the project..." 
 
#  uv venv source .venv/bin/activate uv 
 
#  pip install -r requirements.txt echo ">>> Project built successfully!" echo ""  echo ">>> Configuring git..." 
 
#  git config --global user.email "tetianabass17@gmail.com" git config --global user.name "tanya-bas" git config --global credential.helper store echo ">>> Git configured successfully!" echo ""  echo ">>> Total setup time: $((SECONDS - START_TIME)) seconds"  
#  huggingface-cli login # this only works if you have the token in plain text (please fix this)