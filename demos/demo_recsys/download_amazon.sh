#!/bin/bash

mkdir -p datasets/

for name in 'Baby' 'Digital_Music' 'Office_Products'
do
  wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_${name}_5.json.gz -P datasets/
  wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_${name}.json.gz -P datasets/
done