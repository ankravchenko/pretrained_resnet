#!/bin/bash 

touch cats_recognized.txt
for file in *
do
  python3 test_vgg.py -i "$file" >> cats_recognized.txt
done
