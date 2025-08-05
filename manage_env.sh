#!/bin/bash

ENV_NAME="research_agent_env"

echo "Removing existing environment (if exists)..."
conda deactivate
conda remove --name $ENV_NAME --all -y

echo "Creating new environment..."
conda env create -f environment.yml

echo "Activation instruction:"
echo "Run: conda activate $ENV_NAME"
