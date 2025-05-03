DATA160=~/imagenet-sz/160
DATA352=~/imagenet-sz/352
DATA=~/imagenet

NAME=fgsm_random_4px

CONFIG1=configs/configs_fast_4px_phase1.yml
CONFIG2=configs/configs_fast_4px_phase2.yml
CONFIG3=configs/configs_fast_4px_phase3.yml

PREFIX1=fgsm_random_phase1_${NAME}
PREFIX2=fgsm_random_phase2_${NAME}
PREFIX3=fgsm_random_phase3_${NAME}

OUT1=fgsm_random_train_phase1_${NAME}.out
OUT2=fgsm_random_train_phase2_${NAME}.out
OUT3=fgsm_random_train_phase3_${NAME}.out

END1=output/${PREFIX1}/checkpoint_epoch6.pth.tar
END2=output/${PREFIX2}/checkpoint_epoch12.pth.tar

# training for phase 1
python -u main_fast.py $DATA160 -c $CONFIG1 --output_prefix $PREFIX1 --adv-train fgsm_random | tee $OUT1

# training for phase 2
python -u main_fast.py $DATA352 -c $CONFIG2 --output_prefix $PREFIX2 --resume $END1 --adv-train fgsm_random | tee $OUT2

# training for phase 3
python -u main_fast.py $DATA -c $CONFIG3 --output_prefix $PREFIX3 --resume $END2 --adv-train fgsm_random | tee $OUT3
