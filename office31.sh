#!/bin/bash

gpu=0

for domain in a2d w2d d2w d2a w2a a2w
do
    python pfc_source.py --dset $domain --gpu_id $gpu --office31
    python pfc_target.py --office31 --dset $domain --gpu_id $gpu
done
