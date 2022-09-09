#!/bin/bash
echo "Running isort..."
isort hrl/

echo "Running black..."
black hrl/

echo "Running pylint..."
pylint hrl/