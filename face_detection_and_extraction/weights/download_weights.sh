#!/bin/bash
# install unzip tool in the system first

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1DDFfYF0Iw5HwWcCzIUFqcZ8ds3Ayc7N9" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1DDFfYF0Iw5HwWcCzIUFqcZ8ds3Ayc7N9" -o weights.zip
unzip weights.zip
mv weights/* .
rm weights.zip
rmdir weights

echo Weights Download finished.
