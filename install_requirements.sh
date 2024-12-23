#!/bin/bash

# Upgrade pip to the latest version
python -m pip install --upgrade pip

# Install pip-tools
pip install pip-tools

# Run the update_requirements script
./update_requirements.sh

# Display the contents of the requirements.txt file if it exists
if [ -f "requirements.txt" ]; then
  echo "requirements.txt file content:"
  cat requirements.txt

  # Create a virtual environment if requirements.txt is found
  virtualenv jenv -p python3
  source jenv/bin/activate
  
  # Install the dependencies from requirements.txt
  pip install -r requirements.txt
else
  # Print a message if the requirements.txt file is not found
  echo "requirements.txt file not found"
fi
