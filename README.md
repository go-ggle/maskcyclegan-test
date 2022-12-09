# voco-vc
MaskCycleGAN-VC 모델에서 직접 녹음한 데이터셋을 target화자로 training한 후, test를 진행한 github입니다.\
epoch 6172번 학습하였으며, 사용한 dataset은 vcc2018에서 확인할 수 있습니다.\

MaskCycleGAN-VC 모델이 아닌, Soft-VC 모델 Demo Site를 확인하려면 [여기](https://colab.research.google.com/drive/1b24BkXJYFR_lA8s1zniuR2ypuQg8KJng)를 클릭해주세요.

## Ubuntu 서버에서 모델 Training
1. 먼저, 가상환경을 만들어줍니다.
```
conda create -n 가상환경이름
```
2. 가상환경에 들어간 후, requirements.txt를 다운로드합니다.\
python 3.6.13 version에 맞게 dependency가 맞춰진 requirements.txt 입니다.
```
conda activate 가상환경이름
pip install -r requirements.txt
```
3. voco-vc 파일로 이동한 후, 모델에 맞게 데이터를 전처리합니다.\
먼저, Training에 사용할 데이터를 전처리합니다. speaker_ids에는 전처리가 필요한 speaker의 파일명을 모두 작성해줍니다.
```
python data_preprocessing/preprocess_vcc2018.py \
  --data_directory vcc2018/vcc2018_training \
  --preprocessed_data_directory vcc2018_preprocessed/vcc2018_training \
  --speaker_ids Minjung MinjungF2
```
다음으로, Test에 사용할 데이터를 전처리합니다. speaker_ids에는 전처리가 필요한 speaker의 파일명을 모두 작성해줍니다.
```
python data_preprocessing/preprocess_vcc2018.py \
  --data_directory vcc2018/vcc2018_evaluation \
  --preprocessed_data_directory vcc2018_preprocessed/vcc2018_evaluation \
  --speaker_ids Minjung MinjungF2
```
4. Training 코드를 실행합니다.\
다른 데이터셋으로 training을 진행할 경우, MinjungF2를 Source 화자 파일명으로, Minjung을 Target 화자 파일명으로 모두 대체하여 코드를 실행해줍니다.\
epochs_per_save에는 checkpoint 파일을 저장할 epoch 횟수를 정하여 적어줍니다.\
만약 저장된 checkpoint 파일을 불러와 training을 이이서 하고 싶은 경우, 코드 끝에 --continue_train을 붙여줍니다.
```
python -m mask_cyclegan_vc.train \
    --name mask_cyclegan_vc_MinjungF2_Minjung \
    --seed 0 \
    --save_dir results/ \
    --preprocessed_data_dir vcc2018_preprocessed/vcc2018_training/ \
    --speaker_A_id MinjungF2 \
    --speaker_B_id Minjung \
    --epochs_per_save 100 \
    --epochs_per_plot 10 \
    --num_epochs 6172 \
    --batch_size 1 \
    --decay_after 1e4 \
    --sample_rate 22050 \
    --num_frames 64 \
    --max_mask_len 25 \
    --gpu_ids 0 \
```
## Ubuntu 서버에서 모델 Test
load_epoch에 test하고싶은 epoch 횟수를 적은 다음, 아래의 코드를 실행하여 모델을 Test합니다.\
다른 데이터셋으로 training을 진행한 경우, MinjungF2를 Source 화자 파일명으로, Minjung을 Target 화자 파일명으로 모두 대체하여 코드를 실행해줍니다.\
생성된 오디오는 results/mask_cyclegan_vc_MinjungF2_Minjung/converted_audio 파일 아래에서 확인할 수 있습니다.
```
python -m mask_cyclegan_vc.test \
    --name mask_cyclegan_vc_MinjungF2_Minjung \
    --save_dir results/ \
    --preprocessed_data_dir vcc2018_preprocessed/vcc2018_evaluation \
    --gpu_ids 0 \
    --speaker_A_id MinjungF2 \
    --speaker_B_id Minjung \
    --ckpt_dir /home/ubuntu/MaskCycleGAN-VC/results/mask_cyclegan_vc_MinjungF2_Minjung/ckpts \
    --load_epoch 6172 \
    --model_name generator_A2B \
```
