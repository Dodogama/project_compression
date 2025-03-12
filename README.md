# ece661_vlm_project

image_captioning_project/
│
├── data/
│   ├── __init__.py
│   ├── coco_dataset.py          # COCO dataset loading and preprocessing
│   ├── data_utils.py            # Data handling utilities
│   └── augmentations.py         # Image augmentation functions
│
├── models/
│   ├── __init__.py
│   ├── clip_encoder.py          # CLIP image encoder component
│   ├── gpt2_decoder.py          # DistilGPT2 text generation component
│   ├── caption_model.py         # Combined model architecture
│   └── prompt_engineering.py    # Learnable prompt implementations
│
├── training/
│   ├── __init__.py
│   ├── trainer.py               # Training loop and optimization
│   ├── loss_functions.py        # Custom loss implementations
│   └── config.py                # Training hyperparameters
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py               # BLEU, CIDEr, SPICE implementations
│   ├── evaluator.py             # Evaluation pipeline
│   └── visualizations.py        # Result visualization tools
│
├── inference/
│   ├── __init__.py
│   ├── caption_generator.py     # Inference pipeline
│   └── beam_search.py           # Advanced decoding strategies
│
├── utils/
│   ├── __init__.py
│   ├── logging_utils.py         # Logging functionality
│   └── checkpoint_utils.py      # Model saving and loading
│
├── notebooks/
│   ├── data_exploration.ipynb   # Dataset analysis
│   ├── model_training.ipynb     # Interactive training notebook
│   └── results_analysis.ipynb   # Performance analysis
│
├── scripts/
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── generate_captions.py     # Demo script for inference
│
├── configs/
│   ├── base_config.yaml         # Base configuration
│   ├── prompt_tuning.yaml       # Prompt tuning configuration
│   └── cross_attention.yaml     # Cross-attention configuration
│
├── requirements.txt             # Project dependencies
├── setup.py                     # Package installation
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore file