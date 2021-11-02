#!/bin/bash
# install unzip tool in the system first

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=17FXIcOSaVwvpjsnfenkm1bZNmmG6VBIi" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=17FXIcOSaVwvpjsnfenkm1bZNmmG6VBIi" -o weights.zip
unzip weights.zip
mv weights/* .
rm weights.zip
rmdir weights

echo Weights Download and unzipping finished!
