import os
from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt

# ======== ここを環境に合わせて調整するところ ========

EPOCHS = [17, 18, 20, 21, 22, 48, 49, 50, 51, 52, 100]

TTS_DIR_TEMPLATE = "exp/custom_infer_{epoch}_jvs010_dev/wav"

#ADULT_REF_DIR = "/mnt/c/Users/tarumi.rika/jvs_ver1/jvs010/falset10/wav24kHz16bit"
ADULT_REF_DIR = "/mnt/c/Users/tarumi.rika/Desktop"
INFANT_REF_DIR = "/mnt/c/Users/tarumi.rika/Desktop"
#adult_filename = "BASIC5000_0025" # jvs001_man
adult_filename = "BASIC5000_0152" # jvs014_woman
infant_filename = "sk060_6_0160" # sk(24m)_man
#infant_filename = "sa036_1_0010" # sa(30m)_woman



SR = 24000

N_FFT = 2048
HOP_LENGTH = 300
WIN_LENGTH = 1200
N_MELS = 80
FMIN = 80
FMAX = 7600

# ======== 共通処理関数たち ========

def load_mel(path: str):
    """wav を読み込んで log-mel スペクトログラムを返す (n_mels, T)"""
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
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """同じshapeの配列同士のcosine similarity（flattenして計算）"""
    a_flat = a.flatten()
    b_flat = b.flatten()
    denom = (np.linalg.norm(a_flat) * np.linalg.norm(b_flat)) + 1e-8
    return float(np.dot(a_flat, b_flat) / denom)

def dtw_align_mels(mel_a: np.ndarray, mel_b: np.ndarray, metric: str = "cosine"):
    """
    DTWで (n_mels, T_a) と (n_mels, T_b) をアラインして、
    対応付けに沿って同じ長さに並べた (n_mels, L) と (n_mels, L) を返す。
    """
    # librosa.sequence.dtw は (features, time) でも動きます
    # 戻りの wp は (i,j) のペア列（time index）
    D, wp = librosa.sequence.dtw(X=mel_a, Y=mel_b, metric=metric)

    # wp は終点→始点の順なので反転
    wp = wp[::-1]

    # wp[:,0] が mel_a の time index、wp[:,1] が mel_b の time index
    idx_a = wp[:, 0]
    idx_b = wp[:, 1]

    aligned_a = mel_a[:, idx_a]
    aligned_b = mel_b[:, idx_b]
    return aligned_a, aligned_b

def dtw_cosine_similarity(mel_a: np.ndarray, mel_b: np.ndarray, metric: str = "cosine") -> float:
    """DTWでアライン後にcosine similarity"""
    a_aligned, b_aligned = dtw_align_mels(mel_a, mel_b, metric=metric)
    return cosine_similarity(a_aligned, b_aligned)

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

        # 参照wav（今は固定ファイル）
        adult_path = Path(ADULT_REF_DIR) / f"{adult_filename}.wav"
        infant_path = Path(INFANT_REF_DIR) / f"{infant_filename}.wav"

        for wav_path in tts_wavs:
            uttid = wav_path.stem

            # 成人
            if adult_path.exists():
                try:
                    mel_tts = load_mel(str(wav_path))
                    mel_adult = load_mel(str(adult_path))

                    # DTWで長さ差を吸収してからcos
                    sim_adult = dtw_cosine_similarity(mel_tts, mel_adult, metric="cosine")
                    adult_sims.append(sim_adult)
                except Exception as e:
                    print(f"  [adult] {uttid}: エラー ({e})")

            # 幼児
            if infant_path.exists():
                try:
                    mel_tts = load_mel(str(wav_path))
                    mel_infant = load_mel(str(infant_path))

                    sim_infant = dtw_cosine_similarity(mel_tts, mel_infant, metric="cosine")
                    infant_sims.append(sim_infant)
                except Exception as e:
                    print(f"  [infant] {uttid}: エラー ({e})")

        mean_adult = float(np.mean(adult_sims)) if adult_sims else float("nan")
        mean_infant = float(np.mean(infant_sims)) if infant_sims else float("nan")

        epoch_to_adult_sims.append((epoch, mean_adult))
        epoch_to_infant_sims.append((epoch, mean_infant))

        print(f"  → adult mean sim = {mean_adult:.4f}, infant mean sim = {mean_infant:.4f}")

    epoch_to_adult_sims.sort(key=lambda x: x[0])
    epoch_to_infant_sims.sort(key=lambda x: x[0])

    print("\n=== 平均コサイン類似度（DTWアライン後） ===")
    print("epoch\tadult_sim\tinfant_sim")
    for (e_a, s_a), (e_i, s_i) in zip(epoch_to_adult_sims, epoch_to_infant_sims):
        if e_a != e_i:
            # epochが欠けた場合に備え、ここは安全に
            continue
        print(f"{e_a:>5}\t{s_a:.4f}\t\t{s_i:.4f}")

    epochs = [e for e, _ in epoch_to_adult_sims]
    adult_vals = [s for _, s in epoch_to_adult_sims]
    infant_vals = [s for _, s in epoch_to_infant_sims]

    plt.figure()
    plt.plot(epochs, adult_vals, marker="o", label="adult vs TTS (DTW)")
    plt.plot(epochs, infant_vals, marker="s", label="infant vs TTS (DTW)")
    plt.xlabel("epoch")
    plt.ylabel("cosine similarity")
    plt.title(f"TTS outputs and {adult_filename}/{infant_filename} (DTW-aligned)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"DTWsimilarity_over_epochs_with_{adult_filename}_and_{infant_filename}.png")
    plt.savefig(f"/mnt/c/Users/tarumi.rika/Desktop/DTWsimilarity_over_epochs_with_{adult_filename}_and_{infant_filename}.png")
    print(f"\nDTWsimilarity_over_epochs_with_{adult_filename}_and_{infant_filename}.png を出力しました。")

if __name__ == "__main__":
    main()
