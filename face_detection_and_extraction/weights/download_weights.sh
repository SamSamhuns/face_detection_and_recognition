#!/bin/bash
# install unzip tool in the system first

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1-3BNZQqERxtiAXI2sb_9q2xQK63RHla7" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1-3BNZQqERxtiAXI2sb_9q2xQK63RHla7" -o weights.zip
unzip weights.zip
mv weights/* .
rm weights.zip
rmdir weights

echo Weights Download finished.
