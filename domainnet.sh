#!/bin/bash

gpu=0
beta=2.6

for domain in p2c p2r s2p r2s r2p r2c c2s
do
    python pfc_source.py --dset s2p --domainnet --gpu_id $gpu
    python pfc_target.py --dset $domain --gpu_id $gpu --domainnet --sim_hyper $beta
done
