#!/bin/bash

# attacks=(
#     "apgd"
#     "apgdt"
#     "autoattack"
#     "bim"
#     "cw"
#     "deepfool"
#     "difgsm"
#     "eaden"
#     "eadl1"
#     "eotpgd"
#     "fab"
#     "ffgsm"
#     "fgsm"
#     "gn"
#     "jitter"
#     "jsma"
#     "mifgsm"
#     "nifgsm"
#     "onepixel"
#     "pgd"
#     "pgdl2"
#     "pgdrs"
#     "pgdrsl2"
#     "pifgsm"
#     "pifgsmpp"
#     "pixle"
#     "rfgsm"
#     "sinifgsm"
#     "sparsefool"
#     "spsa"
#     "square"
#     "tifgsm"
#     "tpgd"
#     "upgd"
#     "vmifgsm"
#     "vnifgsm"
# )

# If an argument is provided, split it by comma into the array
if [ "$#" -ge 1 ]; then
    IFS=',' read -r -a attacks <<< "$1"
fi

for attack in "${attacks[@]}"; do
    echo "[*] Running $attack"
    python3 main.py --mode adv_eval --dataset 2016.10a --attack "$attack" --attack_backend torchattacks --dir_name "$attack"
done