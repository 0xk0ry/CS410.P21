# 4eps evaluation
python main_fast.py /kaggle/input/tiny-imagenet-200-resized/original --config configs_fast_4px_evaluate.yml --output_prefix eval_4px --resume /kaggle/working/exp/ImageNet/fast_adv_phase3_eps4_step5_eps4_repeat1/model_best.pth.tar --evaluate --restarts 10

# 2eps evaluation 
python main_fast.py /kaggle/input/tiny-imagenet-200-resized/original --config configs_fast_2px_evaluate.yml --output_prefix eval_2px --resume /kaggle/working/exp/ImageNet/fast_adv_phase3_eps2_step2_eps2_repeat1/model_best.pth.tar --evaluate --restarts 10
