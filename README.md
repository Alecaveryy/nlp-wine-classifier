# NLP Wine Varietal Classifier

> **Portfolio demo – explore the notebook or run it yourself!**

## Overview
Using 130k+ WineMag reviews scraped from Kaggle, this project applies classic NLP and machine‑learning pipelines to turn subjective wine reviews into wine varietal predictions.

## Data
| Source | Rows | Notes |
|--------|------|-------|
| Kaggle “Wine Reviews” (WineMag) | ~130 000 | Free text reviews |

All raw CSVs live in **https://www.kaggle.com/datasets/zynicide/wine-reviews/data**; the project is completely reproducible offline.

## Methods & Libraries
- **Text preprocessing** – tokenisation (`nltk`), stop‑word removal, Count & TF‑IDF vectorisers  
- **Models**  
  - Random Forest Classifier + CountVectorizer  
  - Random Forest Classifier + TF‑IDF  
  - Multinomial Naïve Bayes + CountVectorizer  
  - Multinomial Naïve Bayes + TF‑IDF  
- Evaluation with `scikit‑learn` `classification_report`  
- Visuals via `matplotlib` / `seaborn`

*(See `notebooks/Wine Predictions.ipynb` for full pipeline.)*

## Key Findings
- **Subjective language limits accuracy** – even the best baseline model struggled, highlighting how nuanced tasting notes are.  
- Objective chemical features (acidity, tannin, etc.) could raise performance; a future enhancement might predict price tiers from chemistry + reviewer sentiment.  
- Random Forest with TF‑IDF out‑performed simpler NB baselines, confirming richer term weighting helps.
