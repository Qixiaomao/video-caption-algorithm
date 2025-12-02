video-captioning-transformer/
├─ README.md
├─ pyproject.toml # or setup.cfg + requirements.txt
├─ env.yml # conda env (PyTorch + torchvision + decord/pyav + transformers + evaluate)
├─ configs/
│ ├─ base.yaml
│ ├─ msrvtt_small.yaml
│ └─ msvd_small.yaml
├─ data/
│ ├─ raw/ # original videos/captions
│ ├─ processed/ # extracted frames, JSON annotations
│ └─ cache/
├─ scripts/
│ ├─ download_msvd.sh
│ ├─ extract_frames.py
│ ├─ build_vocab.py
│ └─ bake_small_split.py # create tiny debug split (e.g., 200 vids)
├─ src/
│ ├─ common/
│ │ ├─ registry.py
│ │ ├─ distributed.py
│ │ ├─ utils.py
│ │ └─ metrics.py # BLEU/METEOR/ROUGE/CIDEr/SPICE + retrieval
│ ├─ data/
│ │ ├─ datasets.py # MSVD/MSR-VTT loaders via decord/pyav
│ │ ├─ samplers.py # frame sampling strategies
│ │ └─ collate.py # pad/pack captions; stack frames
│ ├─ models/
│ │ ├─ video_encoder.py # ViT/TimeSformer-lite (from scratch or torchvision ViT)
│ │ ├─ text_decoder.py # TransformerDecoder (causal) with cross-attn
│ │ ├─ captioner.py # VideoCaptioner = encoder + decoder + LM head
│ │ └─ retrieval_head.py # dual-encoder CLIP-style head (optional)
│ ├─ training/
│ │ ├─ loop.py # train/val steps, grad-clip, amp
│ │ ├─ optim.py # AdamW, schedulers (cosine, linear warmup)
│ │ └─ losses.py # XE, label smoothing, SCST hooks
│ ├─ xai/
│ │ ├─ attention_rollout.py
│ │ ├─ grad_cam_vit.py
│ │ └─ token_attribution.py # Integrated Gradients
│ ├─ cli/
│ │ ├─ train.py
│ │ ├─ evaluate.py
│ │ ├─ caption.py
│ │ ├─ retrieve.py
│ │ └─ explain.py
│ └─ viz/
│ ├─ show_caption.html # lightweight HTML viz for attention over frames
│ └─ overlay.py # draw heatmaps on frames
└─ tests/
├─ test_data.py
├─ test_models.py
└─ test_metrics.py