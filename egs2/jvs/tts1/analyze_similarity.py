import os
from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt

# ======== ここを環境に合わせて調整するところ ========

# 解析する epoch のリスト（手元にあるもの全部）
EPOCHS = [17, 18, 20, 21, 22, 48, 49, 50, 51, 52]

# TTS 合成結果のディレクトリパターン（さっき作った infer_epochXX のパス）
TTS_DIR_TEMPLATE = "exp/custom_infer_{epoch}/wav"

# 成人音声（同じ話者・別文）のディレクトリ（★仮のパス）
ADULT_REF_DIR = "/mnt/c/Users/tarumi.rika/jvs_ver1/jvs010/falset10/wav24kHz16bit"

# 幼児音声のディレクトリ（★仮のパス）
INFANT_REF_DIR = "/mnt/c/Users/tarumi.rika/Desktop"

# サンプリング周波数（jvs / TTS と合わせる）
SR = 24000

# fbank と合わせたパラメータ（だいたい conf と揃える）
N_FFT = 2048
HOP_LENGTH = 300
WIN_LENGTH = 1200
N_MELS = 80
FMIN = 80
FMAX = 7600

# ======== 共通処理関数たち ========

def load_mel(path: str):
    """wav を読み込んで log-mel スペクトログラムを返す"""
    y, sr = librosa.load(path, sr=SR)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    # 対数スケール（dB）
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db  # shape: (n_mels, T)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """2 つのベクトルのコサイン類似度（-1〜1）"""
    a_flat = a.flatten()
    b_flat = b.flatten()
    denom = (np.linalg.norm(a_flat) * np.linalg.norm(b_flat)) + 1e-8
    return float(np.dot(a_flat, b_flat) / denom)

def crop_to_min_time(mel_a: np.ndarray, mel_b: np.ndarray):
    """時間長の短いほうに合わせて左右を揃える（簡易版）"""
    T = min(mel_a.shape[1], mel_b.shape[1])
    return mel_a[:, :T], mel_b[:, :T]

# ======== メイン処理 ========

def main():
    epoch_to_adult_sims = []
    epoch_to_infant_sims = []

    print("※ 注意：ADULT_REF_DIR / INFANT_REF_DIR のパスは仮なので、実データに合わせて直してください。")
    print(f"  ADULT_REF_DIR:  {ADULT_REF_DIR}")
    print(f"  INFANT_REF_DIR: {INFANT_REF_DIR}")
    print()

    for epoch in EPOCHS:
        tts_dir = TTS_DIR_TEMPLATE.format(epoch=epoch)
        tts_dir_path = Path(tts_dir)

        if not tts_dir_path.exists():
            print(f"[epoch {epoch}] TTS 生成ディレクトリが見つかりません: {tts_dir}")
            continue

        tts_wavs = sorted(tts_dir_path.glob("*.wav"))
        if not tts_wavs:
            print(f"[epoch {epoch}] {tts_dir} に wav が見つかりません。")
            continue

        adult_sims = []
        infant_sims = []

        print(f"[epoch {epoch}] {len(tts_wavs)} ファイルで類似度を計算中...")

        for wav_path in tts_wavs:
            uttid = wav_path.stem  # 例: jvs010_VOICEACTRESS100_001

            adult_path = Path(ADULT_REF_DIR) / "BASIC5000_1274.wav"
            infant_path = Path(INFANT_REF_DIR) / "sk060_6_0160.wav"

            # 成人音声が無ければスキップ
            if adult_path.exists():
                try:
                    mel_tts = load_mel(str(wav_path))
                    mel_adult = load_mel(str(adult_path))
                    mel_tts_c, mel_adult_c = crop_to_min_time(mel_tts, mel_adult)
                    sim_adult = cosine_similarity(mel_tts_c, mel_adult_c)
                    adult_sims.append(sim_adult)
                except Exception as e:
                    print(f"  [adult] {uttid}: エラー ({e})")
            else:
                # 必要ならここで warn 出す
                pass

            # 幼児音声が無ければスキップ
            if infant_path.exists():
                try:
                    mel_tts = load_mel(str(wav_path))
                    mel_infant = load_mel(str(infant_path))
                    mel_tts_c, mel_infant_c = crop_to_min_time(mel_tts, mel_infant)
                    sim_infant = cosine_similarity(mel_tts_c, mel_infant_c)
                    infant_sims.append(sim_infant)
                except Exception as e:
                    print(f"  [infant] {uttid}: エラー ({e})")
            else:
                # 必要ならここで warn 出す
                pass

        # 平均をとる（1 つも計算できなかった場合は NaN）
        mean_adult = float(np.mean(adult_sims)) if adult_sims else float("nan")
        mean_infant = float(np.mean(infant_sims)) if infant_sims else float("nan")

        epoch_to_adult_sims.append((epoch, mean_adult))
        epoch_to_infant_sims.append((epoch, mean_infant))

        print(f"  → adult mean sim = {mean_adult:.4f}, infant mean sim = {mean_infant:.4f}")

    # 結果をソートしてきれいに表示
    epoch_to_adult_sims.sort(key=lambda x: x[0])
    epoch_to_infant_sims.sort(key=lambda x: x[0])

    print("\n=== 平均コサイン類似度（大きいほど似ている） ===")
    print("epoch\tadult_sim\tinfant_sim")
    for (e_a, s_a), (e_i, s_i) in zip(epoch_to_adult_sims, epoch_to_infant_sims):
        assert e_a == e_i
        print(f"{e_a:>5}\t{s_a:.4f}\t\t{s_i:.4f}")

    # グラフ描画
    epochs = [e for e, _ in epoch_to_adult_sims]
    adult_vals = [s for _, s in epoch_to_adult_sims]
    infant_vals = [s for _, s in epoch_to_infant_sims]

    plt.figure()
    plt.plot(epochs, adult_vals, marker="o", label="adult vs TTS")
    plt.plot(epochs, infant_vals, marker="s", label="infant vs TTS")
    plt.xlabel("epoch")
    plt.ylabel("cosine similarity (log-mel)")
    plt.title("Similarity between TTS outputs and adult/infant speech")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("similarity_over_epochs.png")
    plt.savefig("/mnt/c/Users/tarumi.rika/Desktop/similarity_over_epochs.png")
    print("\nsimilarity_over_epochs.png を出力しました。")

if __name__ == "__main__":
    main()


