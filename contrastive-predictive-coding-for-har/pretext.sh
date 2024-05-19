#!/bin/bash

for dset in 'uci' 'wisdm' 'motionsense' 'kuhar' 'realworld_thigh' 'realworld_waist'
do
    python main.py --dataset $dset
done
