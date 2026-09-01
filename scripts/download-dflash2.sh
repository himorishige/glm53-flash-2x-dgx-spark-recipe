#!/bin/bash
# node2: fetch the DFlash2 drafter (incoai/GLM-5.3-Flash-DFlash2, ~2.2 GB, CC-BY-NC-ND) next to the checkpoint.
# node1 then pulls it over the QSFP rsync daemon:  rsync -a rsync://192.168.200.14:8873/dflash2/ ~/models/GLM-5.3-Flash-DFlash2/
export HF_HUB_DISABLE_XET=1
mkdir -p ~/models/GLM-5.3-Flash-DFlash2 ~/glm53-cluster/logs
echo "DFLASH_START $(date -Is)" >> ~/glm53-cluster/logs/download-dflash2.log
~/.local/bin/hf download incoai/GLM-5.3-Flash-DFlash2 --local-dir ~/models/GLM-5.3-Flash-DFlash2 --max-workers 4 2>&1 | tee -a ~/glm53-cluster/logs/download-dflash2.log
echo "DFLASH_DONE rc=${PIPESTATUS[0]} $(date -Is)" >> ~/glm53-cluster/logs/download-dflash2.log
du -sb ~/models/GLM-5.3-Flash-DFlash2 | tee -a ~/glm53-cluster/logs/download-dflash2.log
