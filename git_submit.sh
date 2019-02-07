#!/bin/bash

git add *.py 
git add *.sh
git add *.ipynb
git commit -m "$1"
git push
