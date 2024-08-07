#python train_vq.py --batch-size 128 --width 512 --lr 1e-4 --total-iter 100000 --lr-scheduler 200000 --code-dim 512 --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir ./output/output_energy --dataname energy --vq-act relu --quantizer ema_reset2 --exp-name VQVAE --window-size 24 --commit 0.001 --gpu 0 &

#export CUDA_VISIBLE_DEVICES="0"
#deepspeed train_arlm.py --exp-name ARTM --batch-size 4096 --num-layers 1 --embed-dim-gpt 1024 --width 512 --code-dim 512 --nb-code 512 --n-head-gpt 16 --block-size 6 --ff-rate 4 --drop-out-rate 0.1 --resume-pth ./output/output_energy/VQVAE/net_best_ds.pth --vq-name VQVAE --out-dir ./output/output_energy/ --total-iter 62500 --lr-scheduler 150000 --lr 0.0008 --dataname energy --down-t 2 --depth 3 --quantizer ema_reset2 --eval-iter 2500 --print-iter 500 --pkeep 0.9 --dilation-growth-rate 3 --vq-act relu --window-size 24
#


#python train_vq.py --batch-size 64 --width 512 --lr 1e-4 --total-iter 100000 --lr-scheduler 200000 --code-dim 512 --nb-code 256 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir ./output/output_stock --dataname stock --vq-act relu --quantizer ema_reset2 --exp-name VQVAE --window-size 24 --commit 2 --gpu 0 &

export CUDA_VISIBLE_DEVICES="1"
deepspeed  train_arlm.py --exp-name ARTM --batch-size 4096 --num-layers 1  --embed-dim-gpt 1024 --width 512 --code-dim 512 --nb-code 256  --n-head-gpt 16 --block-size 6 --ff-rate 4 --drop-out-rate 0.1 --resume-pth ./output/output_stock/VQVAE/net_best_ds.pth --vq-name VQVAE --out-dir ./output/output_stock/ --total-iter 20000 --lr-scheduler 150000 --lr 0.0008 --dataname stock --down-t 2 --depth 3 --quantizer ema_reset2 --eval-iter 2500 --print-iter 500 --pkeep 0.7 --dilation-growth-rate 3 --vq-act relu --window-size 24  &
