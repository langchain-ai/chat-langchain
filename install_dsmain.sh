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

# this is an ugly/temp hack to remediate the fact that pip install of ds-main
# uninstalls/resintalls a newer version of pydantic, which causes errors in
# chat-langchain... we should work on a cleaner fix, for example:
# - adding ds-main as a dependency in chat-langchain's pyproject.toml
# - upgrading chat-langchain to a newer pydantic version?
# but currently this is not priority, so we will keep this ugly hack
pip install "pydantic==1.10.13"
