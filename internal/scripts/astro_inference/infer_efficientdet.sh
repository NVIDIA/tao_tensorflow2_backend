#!/bin/bash
usage() { echo "$0 usage: infer_efficientdet -e /path/to/exp/dir" && grep " .)\ #" $0; exit 0; }
[ $# -eq 0 ] && usage
# exp="/home/scratch.p3/yuw/tlt3_experiments/astro26/effdet-d5-sampler"
rootdir="/home/projects2_metropolis/datasets/kpi_datasets/People"

while getopts he: flag
do
    case "${flag}" in
        e)
            exp=${OPTARG};;
        h | *)
            usage;;
    esac
done

seqs=(
"20180403_140724"
"20ft_45_degrees_2_4.mp4"
"20ft_45_degrees_4.mp4"
"C0024-QA.m4v"
"C0042_4.mp4"
"C0065-QA.m4v"
"C0083-QA.m4v"
"DSC_0001B"
"DSC_0016E"
"Demo_Icetana_KPI-01_181116"
"GP140121"
"P1000162.MOV"
"P1000328"
"P1450794"
)
for val1 in ${seqs[*]}; do
    echo $val1
    cmd0="cd /workspace/tao-tf2/nvidia_tao_tf2/cv/efficientdet/entrypoint && python efficientdet.py inference -e ${exp}/inference.yaml inference.image_dir=${rootdir}/${val1}/images inference.output_dir=${exp}/results/${val1}"
    eval $cmd0
done