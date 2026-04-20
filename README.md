# 🤖 StudyBuddy AI — Advanced ML Edition

> An AI-powered study assistant for Machine Learning students, built with **Python**, **Streamlit**, and **Groq's free API** running **Llama 3.3 70B**.

---

## 📌 Short Description

**StudyBuddy AI** is an interactive web app with Machine Learning algorithms that helps students learn ML through an AI chat tutor, live concept visualizations, and a hands-on ML model training lab. It uses **Groq's free API** with **Llama 3.3 70B** — one of the fastest and most capable open-source models available — so anyone can run it without a paid subscription or credit card.

---

## ✨ Features

### 💬 AI Chat Tutor
- Ask anything about ML, Deep Learning, NLP, RL, and more
- Powered by **Llama 3.3 70B** via **Groq** (free, blazing fast)
- Multiple learning modes: **Explain & Discuss**, **Quiz Me**, **Code Walkthrough**, **Math Deep-Dive**, **Motivate Me**
- Quick-topic buttons: Backpropagation, Attention, GANs, Transformers, and more
- Built-in motivation support for when studying gets hard

### 📊 Interactive ML Visualizer
Six interactive Plotly charts that bring ML theory to life:
- **Gradient Descent** — live loss surface with adjustable learning rate & steps
- **Bias-Variance Tradeoff** — error curves with optimal complexity marker
- **Activation Functions** — ReLU, Sigmoid, Tanh, Leaky ReLU, GELU
- **Learning Rate Effect** — convergence vs divergence across LR values
- **SVM Decision Boundary** — margin and support vector visualization
- **Regularization (L1 vs L2)** — weight shrinkage across lambda values

### 🔬 ML Models Lab
Train real scikit-learn models live in the browser with tunable hyperparameters:

| Category | Models |
|---|---|
| **Classification** | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN, Naive Bayes, MLP |
| **Regression** | Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, SVR, MLP |
| **Clustering** | K-Means (+ Elbow Method), DBSCAN |
| **Dimensionality Reduction** | PCA, t-SNE |

Each model produces: accuracy/R² metrics, confusion matrix, ROC curve, learning curve, feature importance, and 5-fold cross-validation scores.

### 📝 Quick Reference Card
- Key ML formulas (Gradient Descent, Cross-Entropy, Softmax, Attention)
- Optimizer cheat sheet, architecture reference table
- Reusable Python snippets (PyTorch training loop, cross-validation, attention)
- Common ML interview Q&A

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web UI |
| [Groq API](https://console.groq.com) | Free LLM inference |
| [Llama 3.3 70B](https://groq.com) | AI chat model |
| [scikit-learn](https://scikit-learn.org) | ML models & datasets |
| [Plotly](https://plotly.com/python/) | Interactive charts |
| [NumPy](https://numpy.org) / [Pandas](https://pandas.pydata.org) | Data & math |

---

## 📁 Project Structure

```
Study_buddy/
│
├── app.py              # Main Streamlit app (Chat, Visualizer, Quick Reference)
├── ml_models_tab.py    # ML Models Lab (Classification, Regression, Clustering, DR)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Bidisha03Biswas/Study-Buddy-with-Artificial_Intelligence.git
cd Study-Buddy-with-Artificial_Intelligence
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a free Groq API key
1. Go to [console.groq.com/keys](https://console.groq.com/keys)
2. Sign up with your Google or GitHub account
3. Click **"Create API Key"** and copy it (starts with `gsk_...`)

> ✅ No credit card required &nbsp;|&nbsp; ✅ Free tier &nbsp;|&nbsp; ✅ Super fast inference

### 4. Run the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser, paste your Groq API key in the sidebar, and start learning!

---

## 🎓 ML Topics Covered

Gradient Descent · Backpropagation · Vanishing Gradients · Attention Mechanism · Transformers · CNNs · RNNs & LSTMs · GANs · VAEs · Regularization (L1/L2/Dropout) · Bias-Variance Tradeoff · SVM · Random Forest · XGBoost · PCA · t-SNE · K-Means · DBSCAN · Transfer Learning · Reinforcement Learning · SHAP & LIME · Cross-Validation · Hyperparameter Tuning

---

## 🤝 Contributing

Pull requests are welcome! Feel free to open an issue for bugs or feature requests.

---

<p align="center">Built with ❤️ to make Machine Learning accessible to every student</p>
