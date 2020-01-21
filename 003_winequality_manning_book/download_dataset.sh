#!/bin/bash

wget --recursive --no-parent -nH -q --cut-dirs=2 https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/

rm wine-quality/*index.html*
