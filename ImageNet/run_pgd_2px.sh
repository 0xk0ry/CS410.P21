DATA160=/kaggle/input/tiny-imagenet-200-resized/160
DATA352=/kaggle/input/tiny-imagenet-200-resized/352
DATA=/kaggle/input/tiny-imagenet-200-resized/original

NAME=pgd_2px

CONFIG1=configs/configs_pgd_2px_phase1.yml
CONFIG2=configs/configs_pgd_2px_phase2.yml
CONFIG3=configs/configs_pgd_2px_phase3.yml

PREFIX1=pgd_phase1_${NAME}
PREFIX2=pgd_phase2_${NAME}
PREFIX3=pgd_phase3_${NAME}

OUT1=/kaggle/working/exp/ImageNet/pgd_train_phase1_${NAME}.out
OUT2=/kaggle/working/exp/ImageNet/pgd_train_phase2_${NAME}.out
OUT3=/kaggle/working/exp/ImageNet/pgd_train_phase3_${NAME}.out

END1=/kaggle/working/exp/ImageNet/${PREFIX1}/checkpoint_epoch6.pth.tar
END2=/kaggle/working/exp/ImageNet/${PREFIX2}/checkpoint_epoch12.pth.tar

# training for phase 1
python -u main_fast.py $DATA160 -c $CONFIG1 --output_prefix $PREFIX1 --adv-train pgd | tee $OUT1

# training for phase 2
python -u main_fast.py $DATA352 -c $CONFIG2 --output_prefix $PREFIX2 --resume $END1 --adv-train pgd | tee $OUT2

# training for phase 3
python -u main_fast.py $DATA -c $CONFIG3 --output_prefix $PREFIX3 --resume $END2 --adv-train pgd | tee $OUT3
