echo 'liif-c' &&
python train_liif.py --config configs/train-div2k/ablation/train_edsr-baseline-liif-c.yaml &&
echo 'liif-d' &&
python train_liif.py --config configs/train-div2k/ablation/train_edsr-baseline-liif-d.yaml &&
echo 'liif-e' &&
python train_liif.py --config configs/train-div2k/ablation/train_edsr-baseline-liif-e.yaml &&
echo 'liif-u' &&
python train_liif.py --config configs/train-div2k/ablation/train_edsr-baseline-liif-u.yaml &&
echo 'liif-x2' &&
python train_liif.py --config configs/train-div2k/ablation/train_edsr-baseline-liif-x2.yaml &&
echo 'liif-x3' &&
python train_liif.py --config configs/train-div2k/ablation/train_edsr-baseline-liif-x3.yaml &&
echo 'liif-x4' &&
python train_liif.py --config configs/train-div2k/ablation/train_edsr-baseline-liif-x4.yaml