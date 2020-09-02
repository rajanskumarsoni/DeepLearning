# python sonlast.py --lr 0.01 --batch_size 2 --init 1 --save_dir ./ --epochs 1 --dataAugment 0 --train train.csv --val valid.csv --test test.csv

python train.py --lr 0.001 --batch_size 128 --init 1 --save_dir  ./
--dropout_prob .5 --decode_method  0 --beam_width 1
--epochs 15 --train train.csv --val valid.csv



