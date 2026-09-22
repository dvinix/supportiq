# SupportIQ — Free GPU Fine-Tuning Runbook (Kaggle & Colab)

This runbook gives you exact, step-by-step instructions to train your Qwen LoRA adapter on a **100% Free NVIDIA T4 GPU** (zero cost, zero local GPU required).

---

## 1. What You Need Before Starting
You only need 3 files from this repository:
1. `notebooks/kaggle/train_qlora.ipynb` (The fine-tuning notebook)
2. `data/processed/train.jsonl` (21,497 training dialogues)
3. `data/processed/val.jsonl` (2,687 validation dialogues)

---

## 2. Option A: Running on Kaggle (Recommended — 30 Free GPU Hours/Week)

### Step 1: Create a New Kaggle Notebook
1. Log in to [Kaggle.com](https://www.kaggle.com).
2. Click **+ Create** -> **New Notebook** (top left).

### Step 2: Turn on the Free GPU & Internet
In the right-hand **Notebook Settings** panel:
1. **Accelerator:** Select **GPU T4 x2** (or GPU T4 x1).
2. **Internet:** Toggle **Internet On** (required to download Qwen from Hugging Face).

### Step 3: Import the Notebook
1. In the top menu, click **File** -> **Import Notebook**.
2. Select `notebooks/kaggle/train_qlora.ipynb` from your laptop.

### Step 4: Upload the Dataset
1. In the right-hand panel under **Input**, click **Upload** (or drag and drop):
   - Upload `train.jsonl` (from `data/processed/train.jsonl`).
   - Upload `val.jsonl` (from `data/processed/val.jsonl`).
2. *Tip:* If the files upload to `/kaggle/input/supportiq-data/`, simply update the paths in Cell 3:
   ```python
   train_file = "/kaggle/input/supportiq-data/train.jsonl"
   val_file = "/kaggle/input/supportiq-data/val.jsonl"
   ```

### Step 5: Run the Notebook & Watch It Train!
1. Click **Run All** (or run each cell with `Shift + Enter`).
2. **Cell 1 & 2:** Installs dependencies and verifies the T4 16GB GPU.
3. **Cell 4 & 5:** Downloads Qwen in 4-bit NF4 precision and attaches the LoRA adapter. Notice that **only ~1.5% of parameters are trainable**!
4. **Cell 6:** Starts fine-tuning. You will see the live progress bar:
   ```text
   Step 10/150  |  Loss: 2.341
   Step 50/150  |  Loss: 1.108
   Step 100/150 |  Loss: 0.624
   ```
5. **Cell 7:** Tests the model live on sample customer queries and outputs structured JSON!
6. **Cell 8:** Creates `supportiq_final_adapter.zip` (~50 MB).

### Step 6: Download Your Adapter
1. In the right-hand panel under **Output** -> `/kaggle/working/`:
2. Click the three dots `...` next to `supportiq_final_adapter.zip` and select **Download**.
3. Move this zip file to your SupportIQ repo under:
   `artifacts/qlora_adapter/`

---

## 3. Option B: Running on Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Click **Upload** and upload `notebooks/kaggle/train_qlora.ipynb`.
3. In the menu: **Runtime** -> **Change runtime type** -> select **T4 GPU** -> **Save**.
4. Click the folder icon on the left sidebar and drag-and-drop `train.jsonl` and `val.jsonl`.
5. Run all cells!
6. Download `supportiq_final_adapter.zip` from the files sidebar.

---

## 4. What This Accomplishes for Your Architecture
- You now have a working, reproducible fine-tuning script (`src/supportiq/training/train.py`).
- You have trained and saved the **real LoRA weights** on a free GPU.
- These same weights can now be served in our **FastAPI Docker container** and deployed to **AWS SageMaker**!
