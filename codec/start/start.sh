
#pip install librosa==0.8.0

#sleep 66666666666666666666666666666666666666666666666666666666666666666666

proj_dir=""
cd ${proj_dir}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=0


python -m torch.distributed.launch --nproc_per_node 8 main_launch_vqdp.py \
  --log_dir exp_log --basic_model_config ""

