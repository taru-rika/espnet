#!/usr/bin/env bash
set -euo pipefail

epoch=$1  # 17 / 18 / 20 / 21 / 22 / 45 / 46 / 47 / 48 / 49 のどれか

python3 -m espnet2.bin.tts_inference \
  --ngpu 0 \
  --data_path_and_name_and_type dump/raw/jvs010_dev/text,text,text \
  --key_file keys_jvs010_dev.scp \
  --model_file exp/tts_jvs_exp001/${epoch}epoch.pth \
  --train_config exp/tts_jvs_exp001/config.yaml \
  --output_dir exp/tts_jvs_exp001/infer_epoch${epoch}_jvs010_dev \
  --vocoder_file none \
  --config conf/decode.yaml
