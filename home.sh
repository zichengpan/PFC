#!/bin/bash

gpu=0
beta=1.8
alpha=1.8

for domain in r2c a2c c2a c2p r2a p2a a2r r2p p2c p2r c2r a2p
do
    python pfc_source.py --dset $domain --gpu_id $gpu --home
    python pfc_target.py --home --dset $domain --gpu_id $gpu --sim_hyper $beta --dis_hyper $alpha
done
