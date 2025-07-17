# Blockchain Fraudulent Transaction Detection

## Overview

This project introduces a novel machine learning–integrated blockchain framework to detect fraudulent transactions proactively within the Ethereum ecosystem. Unlike traditional systems where fraud detection occurs post-mining, this solution classifies transactions while they are still in the mempool, thereby conserving computational resources and strengthening blockchain security. By combining blockchain validation processes with predictive ML models, it aims to mitigate risks such as double-spending, Sybil attacks, and theft.

---

## Problem Statement

While blockchain offers immutability and decentralization, it lacks real-time fraud detection mechanisms. Once a transaction is mined and added to a block, it cannot be reversed—even if it is later identified as fraudulent. This project addresses the challenge by using a machine learning pipeline that predicts whether a transaction is malicious during the mempool stage, thereby preventing its addition to the blockchain and avoiding irreversible fraud.

---

## Proposed Methodology

Blockchain-based systems are vulnerable to fraud vectors like double-spending and Sybil attacks due to their open and decentralized architecture. To combat these issues, this project proposes a **two-layered system**:

1. **Blockchain Layer**: Responsible for initiating and validating transactions using traditional cryptographic and consensus mechanisms.
2. **Machine Learning Layer**: Introduces fraud detection at the *Validation Queue* (mempool) stage before transactions proceed to mining.

This approach utilizes ensemble learning models trained on historical blockchain transaction data to flag suspicious activities such as:

* Unusual frequency or volume of transactions
* Transactions involving new or unverified addresses
* Behavior inconsistent with prior activity

### Flow Summary:

* Transaction enters the mempool.
* ML classifiers evaluate the transaction and assign a fraud probability.
* If fraud probability exceeds the threshold, it is discarded before mining.
* Otherwise, it proceeds to block confirmation.

---

## System Architecture

```
[User Transaction] 
     ↓
[Mempool / Validation Queue]
     ↓
[ML Ensemble Classifiers: CatBoost, XGBoost, Random Forest]
     ↓
[Soft Voting Aggregator]
     ↓
┌─────────────┐        ┌──────────────┐
│ Legitimate  │        │ Fraudulent   │
│ → Mining    │        │ → Rejected   │
└─────────────┘        └──────────────┘
```
<img width="1069" height="486" alt="image" src="https://github.com/user-attachments/assets/ea3074ad-d732-4e2a-9254-20247e162695" />

---

## Machine Learning Models

The following models were trained and evaluated for the fraud classification task:

1. **CatBoost**

   * Gradient boosting designed for categorical features
   * Captures complex transaction behaviors
   * Highest accuracy among all models

2. **XGBoost**

   * Regularized gradient boosting
   * Optimized for speed and accuracy
   * Good at detecting subtle fraud patterns

3. **Random Forest**

   * Ensemble of decision trees
   * Handles feature interactions well
   * Robust to overfitting and noisy data

4. **AdaBoost**

   * Boosts weak learners by focusing on misclassifications
   * Effective in high-variance datasets

5. **Decision Tree**

   * Easy to interpret, forms decision paths
   * Used for baseline understanding

6. **K-Nearest Neighbors (KNN)**

   * Instance-based learning model
   * Classifies based on transaction similarity

---

## Model Evaluation

| Classifier          | Accuracy (%) |
| ------------------- | ------------ |
| K-Nearest Neighbors | 96.14        |
| Decision Tree       | 97.30        |
| Random Forest       | 98.47        |
| XGBoost             | 98.57        |
| AdaBoost            | 98.76        |
| **CatBoost**        | **99.57**    |

**Proposed Ensemble Model Accuracy**: **99.10%**

> The final model combines CatBoost, XGBoost, and Random Forest using a **soft voting ensemble**, balancing individual strengths and minimizing false classifications.

---

## Technologies Used

* **Programming Language**: Python 3.10+
* **Libraries**:

  * `pandas`, `numpy` – Data wrangling
  * `scikit-learn` – Baseline models and preprocessing
  * `xgboost`, `catboost`, `lightgbm` – Advanced ML models
  * `matplotlib`, `seaborn` – Data visualization
* **Environment**: Jupyter Notebook, Anaconda

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/srummanf/Fraud-Blockchain-Transaction-Detection.git
   cd blockchain-fraud-detection
   ```

2. Set up a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:

   ```bash
   pip install pandas numpy scikit-learn xgboost catboost matplotlib seaborn
   ```
---

## Understand the Code

* The project is implemented in `Code.ipynb`.
* **Key sections in the notebook**:

  * **Data Loading**: Reads the transaction dataset and inspects features.
  * **Preprocessing**: Cleans irrelevant features, handles class imbalance.
  * **Model Training**: Trains classifiers – KNN, DT, RF, XGB, AdaBoost, CatBoost.
  * **Evaluation**: Compares accuracy of each model.
  * **Ensemble**: Uses Soft VotingClassifier to combine top models.
  * **Inference**: Predicts new transactions for fraud likelihood.

> To explore or modify the logic, open the notebook in Jupyter or VSCode with Jupyter extension.

---

## How to Contribute

We welcome contributions to enhance the fraud detection framework!

1. Fork the repository
2. Create a new branch:

   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes
4. Push to your fork
5. Open a Pull Request (PR)

Suggestions:

* Add new features or model support
* Optimize existing classifiers
* Improve visualizations or explanations
* Clean/refactor code

---

## Dataset

This project uses an Ethereum transaction dataset containing:

* Transaction frequency and gap time
* Unique sender/receiver counts
* Total Ether sent/received
* Contract creation indicators
* Binary fraud labels (0 or 1)

> The dataset was curated for research and simulation purposes.
> If not bundled, please contact the maintainers for access or use Ethereum public datasets.

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 Shaikh Rumman Fardeen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions...
```


## Conclusion

This project demonstrates a proactive and highly accurate framework for blockchain fraud detection by integrating ML with Ethereum-like blockchain behavior. The system not only improves fraud detection rates but also reduces unnecessary mining computations, making blockchain networks more efficient, secure, and trustworthy.

---

## Future Scope

* **Integration with Live Ethereum Testnets** (e.g., Ropsten, Goerli)
* **Expansion to Multi-chain Analysis** (Solana, Polygon, BNB)
* **Smart Contract Security Auditing** using ML
* **Reinforcement Learning** for adaptive fraud threshold tuning
* **Real-time API Service** deployment with Flask or FastAPI





