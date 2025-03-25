# AspectBasedSentimentAnalysis
This project focuses on performing **Aspect-Based Sentiment Analysis (ABSA)** on product reviews using deep learning models such as **BiLSTM with Attention** and **Explainable AI (XAI)** techniques like **SHAP** for interpretation. It provides fine-grained sentiment analysis for specific aspects within a review.

---

## 🎯 Goal

To accurately determine the sentiment (positive, negative, neutral) for each **aspect** mentioned in a customer review using an interpretable deep learning pipeline.

---

## 🔧 Technologies Used

- Python
- PyTorch
- TensorFlow (for comparison)
- NLTK / SpaCy (text preprocessing)
- BiLSTM with Attention
- SHAP (SHapley Additive exPlanations)
- Streamlit (optional: UI for demo)

---

## 📌 Features

- Fine-grained sentiment classification per aspect (e.g., "battery life", "camera", etc.)
- Attention mechanism for better context understanding
- Explainable AI using SHAP for aspect-wise transparency
- Preprocessing pipeline with tokenization, lemmatization, and stopword removal
- Optional: Streamlit-based UI for interactive testing

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ABSA-Deep-Learning.git
cd ABSA-Deep-Learning
2. Install Dependencies
bash
Copy
pip install -r requirements.txt
3. Run Training
bash
Copy
python train.py --dataset data/yelp_reviews.csv --model absa_bilstm
4. Evaluate / Test
bash
Copy
python evaluate.py --model absa_bilstm --data test.csv
5. Run Explainable AI (Optional)
bash
Copy
python explain.py --model absa_bilstm --input "The battery life is great but the screen is dull"
6. (Optional) Launch Streamlit UI
bash
Copy
streamlit run absa_app.py
📂 Dataset Used
Primary Dataset: Yelp Reviews (150k rows)

Augmented with: Emotion-rich reviews and synthetic data

Contains labeled aspects and associated sentiment

📷 Demo Screenshot
Aspect	Sentiment	Confidence
battery life	Positive	0.92
screen	Negative	0.79
📄 Project Structure
cpp
Copy
ABSA-Deep-Learning/
├── data/
│   ├── yelp_reviews.csv
├── models/
│   └── absa_bilstm.pt
├── utils/
│   └── preprocess.py
├── train.py
├── evaluate.py
├── explain.py
├── absa_app.py (optional - UI)
├── requirements.txt
└── README.md
📄 License
This project is licensed under the MIT License.

🙌 Acknowledgements
SHAP for Explainable AI

PyTorch

Yelp Open Dataset
