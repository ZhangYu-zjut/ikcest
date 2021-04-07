#!/bin/bash

echo "Begin to generate result, please wait......"

fid="Y055"
cmd="python main.py --approximate floor --normalize 0 --epochs 200 --data ./data/data/data.txt --sim_mat ./data/adj/adjmat_Y055_voronoi0.30HubNoAF.txt --model CNNRNN_Res --dropout 0.2 --ratio 0.01 --residual_window 4 --save_dir mysave --save_name ${fid}_cnnrnn_res.hhs.w-16.h-1.ratio.0.01.hw-4.pt --horizon 1 --window 9 --gpu 3 --metric 0 --city_name ${fid} --hidRNN 51"
eval $cmd
# 调用process_single.py生成Y055.csv 文件保存路径：./predict/format
cmd="python post_proc_single.py --fid ${fid} --file ./predict/temp/model_${fid}.csv --save ./predict/format/submission_${fid}.csv"
eval $cmd
echo -ne "finish 15%,[>>-----------]\r"

# Y081
fid="Y081"
cmd="python main.py --approximate floor --normalize 0 --epochs 200 --data ./data/data/data.txt --sim_mat ./data/adj/adjmat_Y064_voronoi0.40HubNoAF.txt --model CNNRNN_Res --dropout 0.2 --ratio 0.01 --residual_window 4 --save_dir mysave --save_name ${fid}_cnnrnn_res.hhs.w-16.h-1.ratio.0.01.hw-4.pt --horizon 1 --window 9 --gpu 3 --metric 0 --city_name ${fid} --hidRNN 51"
eval $cmd
cmd="python post_proc_single.py --fid ${fid} --file ./predict/temp/model_${fid}.csv --save ./predict/format/submission_${fid}.csv"
eval $cmd
echo -ne "finish 30%,[>>>>---------]\r"

# T193
fid="T193"
cmd="python main.py --normalize 2 --data ./data/data/data.txt --sim_mat ./data/adj/adjmat_Y159_voronoi0.90HubNoACFIJK.txt --model CNNRNN_Res_relu --dropout 0.4 --ratio 0.01 --residual_window 4 --save_dir mysave  --save_name ${fid}_HubAll_relu_cnnrnn_res.hhs.w-9.h-1.ratio.0.01.hw-4-dp0.4.pt  --horizon 1 --window 9 --gpu 1  --metric 0 --city_name ${fid} --output_fun relu --epochs 500 --lr 0.0001 --hidRNN 80"
eval $cmd
cmd="python post_proc_single.py --fid ${fid} --file ./predict/temp/model_${fid}.csv --save ./predict/format/submission_${fid}.csv"
eval $cmd
echo -ne "finish 45%,[>>>>>>-------]\r"


fid="X161"
cmd="python main.py --normalize 2 --data ./data/data/data.txt --sim_mat ./data/adj/adjmat_Y104_voronoi0.05HubNoACFIJK.txt --model CNNRNN_Res_relu --dropout 0.4 --ratio 0.01 --residual_window 4 --save_dir mysave  --save_name ${fid}_HubAll_relu_cnnrnn_res.hhs.w-9.h-1.ratio.0.01.hw-4-dp0.4.pt  --horizon 1 --window 9 --gpu 1  --metric 0 --city_name ${fid} --output_fun relu --epochs 500 --lr 0.0001 --hidRNN 80"
eval $cmd
cmd="python post_proc_single.py --fid ${fid} --file ./predict/temp/model_${fid}.csv --save ./predict/format/submission_${fid}.csv"
eval $cmd
echo -ne "finish 60%,[>>>>>>>>-----]\r"

# Z135
fid="Z135"
cmd="python Curve_linear_V2.py"
eval $cmd
cmd="python post_proc_single.py --fid ${fid} --save ./predict/format/submission_Z135.csv"
eval $cmd
echo -ne "finish 75%,[>>>>>>>>>>---]\r"

# M211--AR
fid="M211"
cmd="python main.py --normalize 0 --data ./data/data/data.txt --model AR --save_dir mysave  --save_name M211_AR.pt  --horizon 1 --window 6 --gpu 1  --metric 0 --city_name ${fid} --epochs 200"
eval $cmd
cmd="python post_proc_single.py --fid ${fid} --file ./predict/temp/model_${fid}.csv --save ./predict/format/submission_${fid}.csv"
eval $cmd
echo -ne "finish 90%,[>>>>>>>>>>>>-]\r"
echo -ne "finish 100%[>>>>>>>>>>>>>]\r"
update_out_file="./predict/format/submission_res.csv"
cmd="python post_proc_update.py --out_file ${update_out_file}"
eval $cmd


