#!/bin/bash

# Navigate to the project directory
cd "/root/jproject/EKYC_" || {
    echo "Directory not found: /root/jproject/EKYC_"
    exit 1
}

# Remove the existing requirements.txt file if it exists
if [ -f requirements.txt ]; then
    echo "Deleting existing requirements.txt..."
    rm requirements.txt
fi

# Compile a new requirements.txt from requirements.in
if [ -f requirements.in ]; then
    echo "Compiling requirements.txt from requirements.in..."
    echo "Disk space before compilation:"
    df -h
    
    pip-compile requirements.in
    
    echo "Disk space after compilation:"
    df -h

    if [ $? -eq 0 ]; then
        echo "Compilation complete! requirements.txt has been updated."
    else
        echo "Error occurred during requirements compilation."
        exit 1
    fi
else
    echo "requirements.in not found. Cannot compile requirements.txt."
    exit 1
fi
