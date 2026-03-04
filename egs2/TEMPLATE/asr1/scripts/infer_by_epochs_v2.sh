et -euo pipefail

FROM_epoch=$1
TO_epoch=$2

for (( epoch=FROM_epoch; epoch<=TO_epoch; epoch++ ))
do
  MODEL_FILE="exp/tts_jvs_exp001/${epoch}epoch.pth"

  # モデルが存在しなければスキップ
  if [ ! -f "$MODEL_FILE" ]; then
    echo "skip epoch ${epoch} (model not found)"
    continue
  fi

  echo "running epoch ${epoch}"

  python3 -m espnet2.bin.tts_inference \
    --ngpu 0 \
    --data_path_and_name_and_type analysis.txt,text,text \
    --model_file "$MODEL_FILE" \
    --train_config exp/tts_jvs_exp001/config.yaml \
    --output_dir exp/custom_infer_${epoch}_jvs010_dev \
    --vocoder_file none

done
