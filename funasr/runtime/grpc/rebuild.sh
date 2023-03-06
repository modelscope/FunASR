#!/bin/bash

rm cmake -rf
mkdir -p cmake/build

cd cmake/build

cmake ../..
make


echo "Build cmake/build/paraformer_server successfully!"
echo "Let's start the server: cd cmake/build/ && ./paraformer_server"
