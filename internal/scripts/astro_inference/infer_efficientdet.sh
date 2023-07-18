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

generic=(
"0_Degrees_20ft_4.mp4"
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
office=(
"IVA-0009-KPI-05_190916_10ft-30deg.mp4"
"IVA-0009-KPI-05_190916_10ft-45-deg.mp4"
"IVA-0009-KPI-05_190916_10ft-60-deg.mp4"
"IVA-0009-KPI-05_190916_11ft-30-deg.mp4"
"IVA-0009-KPI-05_190916_11ft-45-deg.mp4"
"IVA-0009-KPI-05_190916_11ft-60-deg.mp4"
"IVA-0009-KPI-05_190916_8ft-45-deg.mp4"
"IVA-0009-KPI-05_190916_8ft-60-deg.mp4"
)
maxine=(
"IVA-0011-KPI-01_210601_Gopro1_GH013916_1.MP4"
"IVA-0011-KPI-01_210601_Gopro1_GH013916_2.MP4"
"IVA-0011-KPI-01_210601_Gopro1_GH013916_4.MP4"
"IVA-0011-KPI-01_210601_Gopro1_GH023916_1.MP4"
"IVA-0011-KPI-01_210601_Gopro1_GH023916_2.MP4"
"IVA-0011-KPI-01_210601_Gopro1_GH023916_3.MP4"
"IVA-0011-KPI-01_210601_Gopro1_GH023916_4.MP4"
"IVA-0011-KPI-01_210601_Gopro1_GH023916_5.MP4"
"IVA-0011-KPI-01_210601_GoPro3_GH033911_1.MP4"
"IVA-0011-KPI-01_210601_GoPro3_GH033911_2.MP4"
"IVA-0011-KPI-01_210601_GoPro3_GH033911_3.MP4"
"IVA-0011-KPI-01_210601_GoPro3_GH033911_4.MP4"
"IVA-0011-KPI-01_210601_GoPro3_GH033911_5.MP4"
"IVA-0011-KPI-01_210601_GoPro3_GH033911_6.MP4"
"IVA-0011-KPI-01_210601_GoPro3_GH033911_7.MP4"
)
gopro=(
"GOPR0296.MP4"
"GOPR0297.MP4"
"GOPR0298.MP4"
"GOPR0299.MP4"
"GOPR0300.MP4"
)
lowc=(
"IVA-0009-KPI-11_220320_NVR_ch2_main_20220201180404_20220201190000_1_cut_1.mp4"
"IVA-0009-KPI-11_220320_NVR_ch2_main_20220201180404_20220201190000_1_cut_2.mp4"
"IVA-0009-KPI-11_220320_NVR_ch2_main_20220201180404_20220201190000_1_cut_3.mp4"
"IVA-0009-KPI-11_220320_NVR_ch2_main_20220201180404_20220201190000_2_1.mp4"
"IVA-0009-KPI-11_220320_NVR_ch2_main_20220201180404_20220201190000_3_1.mp4"
"IVA-0009-KPI-11_220320_NVR_ch2_main_20220201180404_20220201190000_4_cut_4.mp4"
"IVA-0009-KPI-11_220320_NVR_ch2_main_20220201180404_20220201190000_4_cut_5.mp4"
"IVA-0009-KPI-11_220320_NVR_ch2_main_20220201180404_20220201190000_4_cut_6.mp4"
"IVA-0009-KPI-11_220320_NVR_ch2_main_20220201180404_20220201190000_4_cut_7.mp4"
"IVA-0009-KPI-11_220320_NVR_ch9_main_20220201182012_20220201190003_1_1.mp4"
)
for val1 in ${maxine[*]}; do
    echo $val1
    cmd0="cd /workspace/tao-tf2/nvidia_tao_tf2/cv/efficientdet/entrypoint && python efficientdet.py inference -e ${exp}/inference.yaml inference.image_dir=${rootdir}/${val1}/images inference.output_dir=${exp}/results/${val1}"
    eval $cmd0
done
