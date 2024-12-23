#!/bin/bash

# Update system and install dependencies for pyenv (if not already installed)
sudo yum update -y
sudo yum install -y gcc zlib-devel bzip2-devel readline-devel sqlite-devel wget curl make \
    gcc-c++ libffi-devel git

# Install pyenv (if it's not already installed)
if ! command -v pyenv &> /dev/null; then
  curl https://pyenv.run | bash
  # Add pyenv to bashrc to enable it automatically
  echo -e '\n# Pyenv initialization' >> ~/.bashrc
  echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
  echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
  echo 'eval "$(pyenv init -)"' >> ~/.bashrc
  source ~/.bashrc
else
  echo "pyenv is already installed"
fi

# Check if Python 3.9.20 is already installed via pyenv, if not install it
if ! pyenv versions | grep -q "3.9.20"; then
  pyenv install 3.9.20
else
  echo "Python 3.9.20 is already installed via pyenv"
fi

# Set Python 3.9.20 as the global version for pyenv
pyenv global 3.9.20

# Install virtualenv (if not already installed)
pip show virtualenv &> /dev/null || pip install virtualenv

# Create a virtual environment named jenv
virtualenv jenv -p python3
source jenv/bin/activate

# Upgrade pip to the latest version in the virtual environment
python -m pip install --upgrade pip

# Install pip-tools
pip install pip-tools

# Run the update_requirements script
./update_requirements.sh

# Display the contents of the requirements.txt file if it exists
if [ -f "requirements.txt" ]; then
  echo "requirements.txt file content:"
  cat requirements.txt
  
  # Install the dependencies from requirements.txt
  pip install -r requirements.txt
else
  # Print a message if the requirements.txt file is not found
  echo "requirements.txt file not found"
fi

