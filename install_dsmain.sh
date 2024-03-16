#!/bin/bash

# Create .ssh directory and set proper permissions
mkdir -p /root/.ssh/
chmod 700 /root/.ssh/

cd /

# Copy the SSH key
cp "/app/secrets/dsmain_ssh_ec2" "/root/.ssh/"

# Remove existing ds-main directory if it exists
if [ -d "ds-main" ]; then 
    rm -r ds-main
fi

# Start ssh-agent, set key permissions, add key, and register GitHub in known_hosts
eval $(ssh-agent -s)
chmod 600 /root/.ssh/dsmain_ssh_ec2
ssh-add /root/.ssh/dsmain_ssh_ec2
ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts

# Clone the repository
git clone git@github.com:cropguard-ai/ds-main.git

# Install the repo
cd ds-main
pip install . -v