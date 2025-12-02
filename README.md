<img width="1455" height="615" alt="image" src="https://github.com/user-attachments/assets/5ddd24c6-db58-4add-b357-78e370278619" />


# Video Captioning System Based on ViT and GPT-2

This repository contains the implementation code for the paper: **"Design and Implementation of a Multi-Modal Video Captioning System Based on ViT and GPT-2"**.

## 📂 Project Structure

- `src/`: Core implementation of the video captioning model.
- `scripts/`: Experimental scripts for data preprocessing and testing.
- `Ui/`: Front-end interface code (Chainlit/React).
- `tools/`: Utility scripts for video frame extraction.

## ⚠️ Note on Pre-trained Models and Datasets
Due to GitHub's file size limits, the **pre-trained model weights (checkpoints)** and **large video datasets** are NOT included in this repository.
- **Checkpoints:** The system relies on ViT and GPT-2 weights.
- **Data:** The training data (e.g., MSVD/MSR-VTT) is excluded.


## Video Captioning System with ViT + GPT-2 and Chainlit Front-end

This repository contains the full implementation of my Master's graduation project:  
**An end-to-end Video Captioning System integrating a ViT-based video encoder, GPT-2 text decoder, and a Chainlit-powered front-end for interactive inference.**

The project provides:
- A complete video captioning model  
- Data preprocessing pipeline  
- Training and evaluation scripts  
- A user-friendly Chainlit demo interface  
- Human evaluation results for caption quality  

---

## 🚀 Features

- **Vision Transformer (ViT) Encoder**
- **GPT-2 Language Decoder**
- **CLIP-style Multimodal Projection Head**
- **Frame-based Video Preprocessing**
- **Chainlit UI for Real-time Caption Generation**
- **Human Evaluation using Fluency / Relevance / Specificity / Overall Preference**
- **Support for MSVD Dataset**

---
#### 📈 Model Architecture

<img width="972" height="540" alt="image" src="https://github.com/user-attachments/assets/050bca5d-2dd7-45d8-9878-990cfc1da56a" />



---




---

🤖 Inference (CLI)

Example:

```c:
python src/cli/inference.py \
    --video_path example.mp4 \
    --checkpoint outputs/checkpoints/best.ckpt \
    --num_frames 16

```


Outputs:

```css:
Generated caption: "A woman is preparing food in the kitchen."
```

---

📊 Human Evaluation

A human evaluation survey was conducted using 4 criteria:

**Fluency,Relevance,Specificity,Overall Preference**

Example summary results:

| Criterion          | Avg. Score |
| ------------------ | ---------- |
| Fluency            | 3.38       |
| Relevance          | 2.63       |
| Specificity        | 3.25       |
| Overall Preference | 4.00       |

---
💬 Chainlit Demo (Front-end)

To launch the interactive UI:
```c:
chainlit run Ui/app_chainlit.py -w
```


Then open the local URL shown in the terminal.

In the UI, you can:

Select an inference engine

Paste a video frame directory path

Generate captions interactively


---
🔍 TODO

- Add support for audio-based captioning

- Extend dataset to MSR-VTT

- Improve Chainlit UI for video uploading

- Add BLEU/ROUGE automatic metrics to the demo

- Deploy model via FastAPI backend

---
### How to run

1. **Install Dependencies:**

```bash:
   pip install -r requirements.txt
```

2. **Clone the repository:**
```c:
1. git clone

2. cd video-captioning-project

3. chainlit run chainlit_app.py -w

```
