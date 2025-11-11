# Emotion Recognition in Drug Reviews

Jupyter notebook for emotion recognition in drug reviews: uses BioBERT/RoBERTa/BERT ensemble for emotion analysis, T5/PEGASUS for summarization, detects side effects, provides drug recommendations. Includes baseline ML models, visualizations, and interactive analysis.

## Features

- **Emotion Analysis**: Multi-model ensemble (BioBERT, RoBERTa, BERT) for detecting 8 core emotions (hope, trust, fear, frustration, joy, sadness, anger, surprise)
- **Text Summarization**: T5 and PEGASUS models for generating concise review summaries
- **Side-Effect Detection**: Heuristic-based extraction of medication side effects from review text
- **Drug Recommendation**: Emotion-aware recommender system matching user reviews to suitable medications
- **Baseline Models**: Logistic Regression and Linear SVM for traditional ML comparison
- **Visualizations**: Emotion distribution plots, confusion matrices, and word clouds
- **Interactive Analysis**: Real-time analysis of user-input reviews with recommendations
- **Dataset Processing**: Handles drug review datasets with cleaning, sampling, and emotion augmentation

## How It Works

This Jupyter notebook provides a comprehensive pipeline for analyzing emotions in pharmaceutical reviews:

1. **Data Preparation**: Loads and cleans drug review datasets, applies sampling for performance
2. **Baseline Modeling**: Trains classical ML models (Logistic Regression, SVM) for drug classification
3. **Emotion Detection**: Uses transformer ensemble to analyze emotional content in reviews
4. **Summarization**: Generates abstractive summaries using T5 and extractive summaries using PEGASUS
5. **Side-Effect Extraction**: Identifies potential medication side effects through keyword matching
6. **Recommendation Engine**: Provides personalized drug recommendations based on emotional similarity
7. **Visualization**: Creates plots for emotion distributions, model performance, and side-effect analysis
8. **Interactive Interface**: Allows users to input custom reviews for real-time analysis and recommendations

The system combines traditional machine learning with modern transformer models for comprehensive drug review analysis.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/srimokshith/Emotion-Recognition.git
cd emotion-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

4. Launch Jupyter notebook:
```bash
jupyter notebook emotion-recognition.ipynb
```

## Requirements

- Python 3.8+
- PyTorch (with CUDA support recommended for GPU acceleration)
- Transformers 4.44.2+
- SpaCy 3.6.1+ with en_core_web_sm model
- Datasets 2.19.2+
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- WordCloud
- Tqdm

## Usage

1. **Data Loading**: The notebook loads drug review datasets (expected format: CSV with review text, ratings, drug names, conditions)
2. **Model Training**: Run baseline models for drug classification comparison
3. **Emotion Analysis**: Process reviews through the emotion ensemble for emotional content analysis
4. **Summarization**: Generate summaries using T5 and PEGASUS models
5. **Recommendations**: Use the emotion-aware recommender for personalized drug suggestions
6. **Visualization**: Explore emotion distributions, model performance, and side-effect word clouds
7. **Interactive Mode**: Input custom reviews for real-time analysis and recommendations

### Example Usage:
```python
from emotion_core_full_fast import EmotionEnsemble, EmotionAwareRecommender

# Load emotion model
emotion_model = EmotionEnsemble()

# Analyze a review
review = "This medication helped my anxiety but caused dizziness."
emotions = emotion_model.analyze_emotions(review)
print(emotions)
```

## Configuration

- **Emotion Models**: BioBERT, RoBERTa, BERT (customizable weights)
- **Summarization Models**: T5-small, PEGASUS-xsum
- **Dataset Sampling**: Adjustable sample size for performance vs. coverage trade-off
- **Emotion Thresholds**: Configurable emotion detection sensitivity
- **Recommendation Parameters**: Content vs. emotion weight balancing

## Dataset

The notebook is designed to work with drug review datasets containing:
- Review text
- Drug names
- Medical conditions
- User ratings
- Review usefulness counts

Expected input format: CSV file with columns for patient_id, drugName, condition, review, rating, etc.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Disclaimer

See [DISCLAIMER.md](DISCLAIMER.md) for important usage notices and limitations.

## Contact

For questions or contributions:
- **Phone**: +91 9392597727
- **Email**: srimokshithinturi@gmail.com

## Contributing

Contributions are welcome! Please read the disclaimer and ensure your code follows best practices for responsible AI development.
