from espnet2.bin.tts_inference import Text2Speech
import soundfile as sf

model_id = "espnet/kan-bayashi_jvs_jvs010_vits_prosody"

tts = Text2Speech.from_pretrained(model_id)

analysis_file = "analysis.txt"

with open(analysis_file, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # key と text を分割
        key, text = line.split(maxsplit=1)

        # 音声合成
        output = tts(text)
        wav = output["wav"]

        # key をファイル名にする
        filename = f"./completeTTS_out/{key}.wav"

        sf.write(filename, wav.cpu().numpy(), tts.fs)

        print(f"saved {filename}")
