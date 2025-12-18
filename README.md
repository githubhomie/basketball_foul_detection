# Basketball Foul Detection

We built a system to automatically detect and classify fouls in NBA broadcast footage. Given a video clip, the model identifies if/when a foul occurs and what type it is.

## What We Did

**The Problem:** Foul detection in basketball is subjective and fast - refs make split-second calls that affect game outcomes. We wanted to see if a neural network could learn to spot fouls from broadcast video alone.

**Our Approach:**
1. **Built a dataset from scratch** - We pulled play-by-play data from the NBA API to find foul events, downloaded 7.5-second clips around each foul, and manually annotated the exact frame where contact occurred.
2. **Created an annotation tool** - A Streamlit web app that streams frames from S3 and lets annotators mark the precise foul moment. We annotated 1,400+ foul clips this way.
3. **Adapted E2E-Spot for basketball** - E2E-Spot was designed for soccer action spotting. We modified it for our foul detection task, experimenting with different backbones, temporal models, and loss function parameters.
4. **Trained on AWS/Colab** - Full training pipeline with Weights & Biases logging, checkpoint management, and evaluation scripts.

**Dataset Stats:**
- 2,360 total clips (1,359 fouls + 1,000 non-fouls)
- 5 foul types: shooting, personal, loose ball, charging, offensive
- ~82,000 frames from 2023-24 NBA season

## Running the Code

**Training (Colab):**
Open `nba_foul_e2e_spot_colab.ipynb` in Google Colab and run all cells. The notebook handles data download, training, and evaluation.

**Data Collection:**
```bash
python data_pipeline/collect_data.py --season 2023-24 --games 100
```

**Annotation Tool:**
```bash
cd data_pipeline/annotation_tool && streamlit run app.py
```

## Team Contributions

**Tianli:** Dataset design and collection, foul video ingestion, annotation coordination, training experiments.

**Oliver:** Model development and training, E2E-Spot adaptation, experiment tracking, hyperparameter tuning.

**Kai:** Evaluation design, detection/classification metrics, threshold tuning, dataset validation.

**Alexandra:** Annotation tooling (Streamlit interface), dataset curation, error analysis, result interpretation.

## Architecture

We use E2E-Spot (Hong et al., ECCV 2022) - a temporal action spotting model with:
- RegNet-Y backbone with Gated Shift Module for efficient video features
- Bidirectional GRU for temporal modeling
- Per-frame classification with focal loss

Key hyperparameters we tuned:
- `dilate_len=3` (±0.75 sec tolerance for annotation variance)
- `fg_upsample=1.0` (oversample foul frames in training)
- `rny008_gsm` backbone (4x larger than baseline)

## Repository Structure

```
basketball_foul_detection/
├── nba_foul_e2e_spot_colab.ipynb   # Main training notebook (run in Colab)
│
├── data_pipeline/                   # [OUR WORK] NBA data collection
│   ├── annotation_tool/             # Streamlit app for labeling fouls
│   ├── collect_data.py              # Pull foul clips from NBA API
│   └── prepare_for_training.py      # Convert to E2E-Spot format
│
├── aws_training/                    # [OUR WORK] AWS/EC2 training scripts
├── data/basketball/                 # [OUR WORK] Dataset configs (train/val/test splits)
│
├── model/                           # [E2E-SPOT] Model architecture
├── dataset/                         # [E2E-SPOT] Dataset loading
├── util/                            # [E2E-SPOT] Utilities
├── evaluation/                      # [E2E-SPOT] Evaluation metrics
└── notebooks/archive/               # Older experiment notebooks
```

Base model code from [E2E-Spot](https://github.com/jhong93/spot) (Hong et al., ECCV 2022).
