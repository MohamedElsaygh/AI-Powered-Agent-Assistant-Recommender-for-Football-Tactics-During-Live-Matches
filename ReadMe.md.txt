# ⚽ AI Agent and Recommender for Football Tactics
### MSc Dissertation Project – University of Sussex  
**Author:** Mohamed Elsaygh  
**Supervisor:** Prof. Thomas Nowotny  
**External Supervisor:** Nigel Jacklin  
**Degree:** MSc Artificial Intelligence and Adaptive Systems (Distinction, First-Class Honours)  

---

## 🎯 Project Overview

This project presents a **real-time AI Agent and Tactical Recommender for Football**, developed as part of my MSc dissertation.  
The system predicts and explains **live substitution decisions** using **machine learning**, **contextual match data**, and **explainable AI (XAI)** methods.

The goal: help coaches make **data-driven, context-aware tactical decisions** during live football matches.

---

## ⚙️ Core System Features

- **Real-time substitution prediction** using contextual data (scoreline, match phase, stamina, and positional metrics)
- **Four predictive pipelines** addressing different tactical questions:
  1. **Pipeline A:** Predicts whether a substitution will occur.
  2. **Pipeline B:** Recommends who *should* be subbed (fatigue or low impact).
  3. **Pipeline C:** Classifies substitution intent (offensive vs defensive).
  4. **Pipeline D:** Detects when *not* to substitute (risk-averse gate).

- **Models:** Logistic Regression, SVM, Random Forest, ANN, XGBoost, and Stacked Ensemble  
- **Validation:** Leave-One-Team-Out Cross Validation (LOGO-CV) for real-world robustness  
- **Explainability:** SHAP values to highlight key tactical features influencing each decision  
- **Calibration:** Threshold tuning + PR-AUC optimization for operational deployment  

---

## 🧠 Data and Feature Engineering

- Based on **StatsBomb Open Data** (event + lineup data)
- Real-time computable features:
  - Player stamina estimation  
  - Score margin dynamics  
  - Time-window context (last 15 minutes)  
  - Team momentum metrics  
  - Positional and role encoding  

---

## 📊 Key Results

- **Context-aware models** significantly outperformed basic time-based predictions.  
- **XGBoost** and **Stacked Ensemble** achieved top performance across accuracy, macro-F1, and PR-AUC.  
- SHAP analysis revealed **score margin**, **time left**, and **stamina** as dominant tactical predictors.  
- Demonstrated that AI can transparently augment **live coaching decisions**.  

---

## 🔍 Explainability & Ethics

- SHAP-based transparency ensures every prediction is **interpretable** and **trustworthy**.  
- Designed for coach-facing applications with **human-in-the-loop decision support**.  
- Ethical considerations include bias mitigation, explainability, and responsible deployment in professional football settings.

---

## 🧩 Future Work

- Integrating **player tracking and computer vision** data for richer context  
- Expanding to **offensive/defensive tactical labelling** and **policy learning**  
- Real-time **agent-based reinforcement learning** for adaptive tactics  
- Partnership with professional clubs for **in-field validation**  

---

## 🏆 Academic Impact

- Completed with **Distinction (First-Class Honours)**  
- Supervised by **Prof. Thomas Nowotny** and **Nigel Jacklin**  
- Forms the foundation of an upcoming **AI-driven football tactics startup**  

---

## 📬 Contact

- **Email:** mohamedelsaygh@gmail.com  
- **LinkedIn:** [linkedin.com/in/mohamedelsaygh](https://linkedin.com/in/mohamedelsaygh)  
- **GitHub:** [github.com/MohamedElsaygh](https://github.com/MohamedElsaygh)

---

> “The future of football isn’t just data — it’s intelligence that adapts.”
