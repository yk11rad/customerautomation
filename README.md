# customerautomation
## Overview
This Google Colab notebook implements an advanced **Customer Support Automation System** for an e-commerce platform, featuring a **Chatbot** and **Automated FAQ System**. It uses **spaCy** for NLP, enhanced with vectorized similarity, dynamic thresholding, and conversation state management. The system simulates 1,000 queries to compare **automated** vs. **manual** support, measuring **response time**, **accuracy**, and **cost**.

### Key Features
- **FAQ System**: Matches queries to 10 FAQs using vectorized cosine similarity.
- **Chatbot**:
  - Dynamic thresholds by priority (low: 0.65, medium: 0.7, high: 0.75).
  - Tracks conversation state for follow-ups and negative feedback.
  - Escalates urgent/low-confidence queries to humans.
- **Realism**:
  - Queries: 80% paraphrased FAQs (via `nlpaug`), 20% unrelated.
  - Priorities (low, medium, high) affect thresholds and manual times.
  - Robust error handling ensures consistent outputs.
- **Optimizations**:
  - Vectorized similarity (~10x faster).
  - Cached embeddings (`lru_cache`).
  - Logging for real-time analytics (CSV-based).
- **Metrics**:
  - **Response Time**: Automated (~0.1s) vs. manual (~300s).
  - **Accuracy**: Automated (similarity-based) vs. manual (90–95%).
  - **Cost**: Automated ($0.01/query) vs. manual ($1/query).
  - **Evaluation**: Confusion matrix, precision, recall, F1-score.

## Requirements
- Google Colab environment.
- Libraries: `spacy` (with `en_core_web_md`), `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `nlpaug`, `nltk`.
- Run the first cells to install dependencies and download NLTK resources (`averaged_perceptron_tagger_eng`, `wordnet`).

## How to Run
1. Open in Google Colab: https://colab.research.google.com/
2. Copy and paste the code into a new notebook.
3. Run all cells (`Ctrl+F9`).
4. Outputs include:
   - **Console**:
     - Efficiency metrics (time, accuracy, cost).
     - Confusion matrix and classification report.
   - **Plots**:
     - Response time histogram.
     - Accuracy and cost bar plots.
     - Response time by priority (boxplot).
   - **CSV**: `metrics_log.csv` for analytics.
   - **Insights**: Automation reduces time/cost by ~99%, with accuracy nearing human levels.

## Customization
- Add FAQs to `faq_data`.
- Adjust `THRESHOLDS` or `n_queries`.
- Modify `aug` for paraphrase variety.
- Extend escalation logic with real CRM integration.

## Notes
- Random seed (`np.random.seed(42)`) ensures reproducibility.
- Accuracy depends on paraphrase quality; tune `nlpaug` settings.
- Dash was replaced with CSV logging due to Colab limitations.
- Manual times assume human agents; adjust for realism.
- Fixed `KeyError: similarity` by ensuring consistent response dictionaries.
- NLTK resources support `nlpaug` paraphrasing.

## Author
Edward Antwi. For further information, please contact me on linkedin at https://www.linkedin.com/in/edward-antwi-8a01a1196/
