# FasterWhisper 批次字幕產生工具

這個專案會讀取 `Videos/` 內的影音檔，使用
[SoybeanMilk/faster-whisper-Breeze-ASR-25](https://huggingface.co/SoybeanMilk/faster-whisper-Breeze-ASR-25)
進行中文語音辨識，並同時輸出：

- `SRT/<檔名>.srt`：含時間軸的字幕檔
- `TEXTONLY/<檔名>.txt`：只有辨識文字的純文字檔

目前支援的輸入副檔名為 `.mp4`、`.mkv`、`.m4a`。

## 已測試環境

- Ubuntu 22.04
- NVIDIA RTX 4090
- CUDA 12.6、cuDNN 9.5
- Python 3.11.10
- faster-whisper 1.1.0
- CTranslate2 4.5.0

程式預設使用 NVIDIA GPU、CUDA 與 FP16。若電腦沒有 NVIDIA GPU，請參考下方的「改用 CPU」說明。

## 專案結構

```text
FasterWhisper/
├── 01_run_python.sh     # 執行程式
├── ca_env.sh            # 啟用 conda 環境
├── main.py              # 主程式及設定
├── requirements.txt     # Python 套件版本
├── Videos/              # 放入待辨識的影音檔
├── SRT/                 # 產生的 SRT 字幕
└── TEXTONLY/            # 產生的純文字逐字稿
```

模型不包含在這個專案資料夾中。每位使用者第一次執行時，會從 Hugging Face 下載到自己的快取目錄。

## 1. 建立環境

先安裝 [Miniconda 或 Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)，再進入專案目錄執行：

```bash
conda create -n faster-whisper python=3.11 -y
conda activate faster-whisper
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

之後每次開啟新的終端機，可以直接執行：

```bash
cd /你的路徑/FasterWhisper
source ca_env.sh
```

`ca_env.sh` 必須用 `source`，不能使用 `./ca_env.sh`；否則 conda 環境不會保留在目前的 shell。

如果剛安裝完 conda，但目前的 bash 尚未載入它，可依實際安裝位置執行其中一個指令，再初始化並重開終端機：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
# 或：source ~/anaconda3/etc/profile.d/conda.sh
conda init bash
```

### GPU 必要條件

預設設定需要：

- 可正常使用的 NVIDIA 驅動程式
- CUDA 12 runtime
- cuBLAS for CUDA 12
- cuDNN 9 for CUDA 12

可先用以下指令確認驅動及 GPU 是否可見：

```bash
nvidia-smi
```

不同作業系統的 CUDA 安裝方式不同，請參考
[faster-whisper 官方 GPU 說明](https://github.com/SYSTRAN/faster-whisper#gpu)。
本專案不需要安裝 PyTorch；影音解碼由 faster-whisper 使用的 PyAV 處理，通常也不必另外安裝系統版 FFmpeg。

## 2. 下載模型

### 方法 A：首次執行時自動下載（最簡單）

不需要先做任何事。`main.py` 執行到 `WhisperModel(...)` 時會自動下載：

```text
SoybeanMilk/faster-whisper-Breeze-ASR-25
```

模型約 3.09 GB，第一次載入需要一些時間；下載完成後會使用本機快取，不會每次重新下載。Hugging Face 預設快取通常位於 `~/.cache/huggingface/`。

看到下面的訊息不是錯誤，公開模型仍可正常下載：

```text
Warning: You are sending unauthenticated requests to the HF Hub.
```

若希望提高下載限額，可先建立免費的 Hugging Face 帳號及 Access Token，然後登入：

```bash
hf auth login
```

不要把 Token 寫進程式、README，或分享給其他人。

### 方法 B：先手動下載到 Hugging Face 快取

```bash
hf download SoybeanMilk/faster-whisper-Breeze-ASR-25
```

下載完成後照常執行程式即可；`main.py` 會找到快取中的模型。

若要把快取放在指定磁碟，下載與執行程式時必須使用相同的 `HF_HOME`：

```bash
export HF_HOME=/你的模型快取路徑/huggingface
hf download SoybeanMilk/faster-whisper-Breeze-ASR-25
./01_run_python.sh
```

### 方法 C：下載成可搬移的本機模型資料夾

```bash
mkdir -p models
hf download SoybeanMilk/faster-whisper-Breeze-ASR-25 \
  --local-dir models/faster-whisper-Breeze-ASR-25
```

接著把 `main.py` 中的模型設定改成：

```python
MODEL_NAME = str(BASE_DIR / "models" / "faster-whisper-Breeze-ASR-25")
```

這種方式適合先在可連網的電腦下載，再將整個 `models/faster-whisper-Breeze-ASR-25/` 搬到離線電腦。模型很大，複製或打包前請先確認磁碟空間。

## 3. 使用方式

1. 把要辨識的 `.mp4`、`.mkv` 或 `.m4a` 檔案放進 `Videos/`。
2. 啟用環境並執行程式：

```bash
source ca_env.sh
./01_run_python.sh
```

也可以直接執行：

```bash
python main.py
```

若腳本沒有執行權限：

```bash
chmod +x 01_run_python.sh
```

程式會依檔名排序並逐一處理所有影音檔。例如：

```text
Videos/demo.mp4
  -> SRT/demo.srt
  -> TEXTONLY/demo.txt
```

處理過程會在終端機即時印出辨識結果。若輸出檔已存在，程式會覆寫該檔案。

## 4. 常用設定

設定集中在 `main.py` 開頭：

```python
MODEL_NAME = "SoybeanMilk/faster-whisper-Breeze-ASR-25"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
LANGUAGE = "zh"
BEAM_SIZE = 5
```

- `LANGUAGE = "zh"`：辨識中文。
- `BEAM_SIZE` 越高通常搜尋越完整，但速度較慢。
- `vad_filter=True`：過濾較長的無語音區段。

### 改用 CPU

沒有可用的 NVIDIA GPU 時，把 `main.py` 改成：

```python
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
```

CPU 可以執行，但這個大型模型通常會比 GPU 慢很多。

## 5. 常見問題

### 第一次停在 `[INFO] Loading model...`

通常正在下載約 3.09 GB 的模型，請先等待並檢查網路與磁碟空間。未登入 Hugging Face 時顯示 unauthenticated warning 仍屬正常。

### 不小心按了 `Ctrl+Z`

`Ctrl+Z` 只是暫停程序，不是正常結束。可以用：

```bash
jobs
fg
```

把程序切回前景繼續執行。若要終止，切回前景後按 `Ctrl+C`。

暫停中的程序可能仍持有 Hugging Face 模型快取的下載鎖。若未先繼續或終止舊程序就再次執行，新的程序可能一直停在 `[INFO] Loading model...` 等待鎖。

### 模型下載速度非常慢

若 Hugging Face 的 Xet 下載路徑長時間沒有進度，可先依上方方式正常終止舊程序，再停用 Xet、改用一般 HTTP range 續傳：

```bash
HF_HUB_DISABLE_XET=1 ./01_run_python.sh
```

現有的 `.incomplete` 暫存檔會保留，通常可以接續下載，不必從頭開始。這個環境變數只需用在下載模型時；模型完整進入快取後，平常執行不需要再設定。

### 出現 CUDA、cuDNN 或 `libcublas` 錯誤

先確認 `nvidia-smi` 正常，再確認 CUDA 12、cuBLAS 及 cuDNN 9 的 runtime library 能被目前 shell 找到。若只需要先測試流程，也可以依上方設定改用 CPU。

### 顯示 `[WARN] No videos found.`

請確認檔案放在專案內的 `Videos/`，且副檔名是 `.mp4`、`.mkv` 或 `.m4a`。

## 補充：搬移專案

專案搬移到其他路徑不影響執行，因為程式會以 `main.py` 所在的位置當作專案根目錄。只要完整複製本專案、重新建立 conda 環境，並讓目前使用者對專案目錄有讀寫權限即可。

Hugging Face 模型快取通常屬於原使用者，不會跟著專案資料夾一起複製；新使用者必須自行下載模型，或使用上方「方法 C」一起搬移本機模型資料夾。

## 分享前注意

打包或上傳專案前，請先檢查 `Videos/`、`SRT/` 與 `TEXTONLY/` 是否含有不應分享的影片、字幕或逐字稿。一般只需分享程式、腳本、README、`requirements.txt`，以及三個空的輸入／輸出目錄；採用自動下載時，不必附上約 3.09 GB 的模型。

Hugging Face Token 儲存在個人環境中即可，不要把 Token、shell 設定檔或 Hugging Face 快取打包給其他人。

## License

本專案自行撰寫的程式碼採用 [MIT License](LICENSE)。

本 repository 不包含模型權重或第三方 Python 套件。它們會由使用者另外下載，並適用各自的授權：

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)：MIT License
- [SoybeanMilk/faster-whisper-Breeze-ASR-25](https://huggingface.co/SoybeanMilk/faster-whisper-Breeze-ASR-25)：Apache License 2.0
- [MediaTek-Research/Breeze-ASR-25](https://huggingface.co/MediaTek-Research/Breeze-ASR-25)：Apache License 2.0

## 參考資料

- [faster-whisper 官方專案](https://github.com/SYSTRAN/faster-whisper)
- [Breeze-ASR-25 的 CTranslate2 模型](https://huggingface.co/SoybeanMilk/faster-whisper-Breeze-ASR-25)
- [原始 MediaTek Breeze-ASR-25 模型](https://huggingface.co/MediaTek-Research/Breeze-ASR-25)
