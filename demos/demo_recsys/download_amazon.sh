#!/bin/bash

mkdir -p local/data/demo_recsys/
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Baby_5.json.gz -P local/data/demo_recsys/
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Baby.json.gz -P local/data/demo_recsys/