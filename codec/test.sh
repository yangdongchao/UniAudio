
run_dir="model_ckpts"

ckpt="$(ls -dt "${run_dir}"/*.pth | head -1 || true)"
ckpt_name=$( basename ${ckpt} )
#bw=1.5
bw=6
outdir=$( dirname $( dirname ${ckpt} ))/output_${ckpt_name%.*}_bw${bw}kpbs_$(date '+%Y-%m-%d-%H-%M-%S')
echo Output in $outdir

wav_dir=""

python inference.py ${wav_dir} \
  ${outdir} --resume_path ${ckpt} --bw ${bw}
