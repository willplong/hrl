#!/bin/bash
echo "Running isort..."
isort hrl/

echo "Running black..."
black hrl/

echo "Running flake8..."
flake8 hrl/

# Do not run pylint during development
# echo "Running pylint..."
# pylint hrl/

echo "Stripping notebook output..."
nbstripout --drop-empty-cells notebooks/*
