# 🎓 Viva Voce Examination: Top 10 Questions & Answers

### 1. What is the fundamental mathematical difference between Bag-of-Words (BoW) and TF-IDF?
**Answer:**
- **Bag-of-Words ($X_{ij} = c(w_j, d_i)$)** measures raw token occurrence counts within a document. It treats all vocabulary words with equal intrinsic importance regardless of whether the word appears everywhere or only in specific documents. Longer texts naturally generate larger vector magnitudes.
- **TF-IDF ($\text{TF-IDF} = \text{TF} \times \log\left(\frac{1 + N}{1 + \text{DF}}\right) + 1$)** penalizes ubiquitous terms that appear across almost all documents (e.g. conversational stopwords, generic business phrases) while heavily weighting rare, highly discriminative tokens (e.g. `lottery`, `urgent`, `guaranteed`, `claim`). Furthermore, L2-norm normalization ensures invariance to message character and word length.

---

### 2. Why is Precision on the Spam class more crucial than Recall in real-world spam detection?
**Answer:**
- **False Positive (FP):** Classifying a legitimate personal or business email/SMS as spam causes catastrophic disruptions (e.g. missing an urgent job interview offer, flight gate change, or medical emergency notification).
- **False Negative (FN):** Letting an occasional spam message slip into the inbox is a minor annoyance that the user can manually delete.
- Therefore, in email and SMS spam detection, we prioritize models with **$\approx 100\%$ Precision** (such as Linear SVM or SVM RBF) to minimize False Positives, while maximizing Spam Recall as a secondary objective.

---

### 3. How did you prevent data leakage during feature extraction and class imbalance handling?
**Answer:**
1. **Stratified Split First:** The dataset was partitioned into 70% Train, 15% Validation, and 15% Test sets *before* any feature extraction occurred.
2. **Train-Only Vectorizer Fitting:** Both `CountVectorizer` and `TfidfVectorizer` were fitted strictly on `X_train`. Validation and Test corpora were transformed using `transform()` without refitting. This guarantees out-of-fold vocabulary distributions never influence the feature representations.
3. **Train-Only SMOTE / Resampling:** Oversampling techniques (like SMOTE) and class weight calculations were derived strictly from the training partition to ensure synthetic points never bleed into test evaluations.

---

### 4. Why did SVM (RBF Kernel) and Linear SVM outperform Naive Bayes on TF-IDF features?
**Answer:**
- Multinomial Naive Bayes relies on the strong **conditional independence assumption** between words, and its generative distribution assumes integer counts (which is natively aligned with Bag-of-Words). When applied to continuous, normalized TF-IDF vectors, its probabilistic formulation can be sub-optimal.
- Support Vector Machines (LinearSVC & RBF SVC) are **discriminative margin-maximizers**. In high-dimensional sparse text spaces ($D \ge 5000$), SVMs find an optimal hyper-plane that separates classes with maximal margin, naturally handling correlated bigrams (e.g., `cash prize`, `account suspended`) and nonlinear decision boundaries without independence assumptions.

---

### 5. Why did you implement a 70/15/15 split instead of a standard 80/20 train/test split?
**Answer:**
A three-way stratified split provides:
- **70% Training:** Sufficient training volume to capture linguistic patterns and diverse spam vocabulary.
- **15% Validation:** Dedicated partition for hyperparameter tuning (GridSearchCV grid search, $C$, $\alpha$, kernel gamma) and model architecture selection.
- **15% Held-Out Test Set:** An untouched, unbiased benchmark partition used exclusively to report final generalization metrics, confusion matrices, and ROC curves, ensuring that hyperparameter tuning does not overfit the reported evaluation test set.

---

### 6. What extra engineered features were extracted and why?
**Answer:**
Spam and phishing attacks exhibit distinct structural characteristics beyond standard vocabulary:
- **URL Count:** Spam messages frequently include shortened links or malicious domains.
- **Email Count & Phone Number Count:** Call scams contain explicit callback telephone numbers or contact emails.
- **Uppercase Ratio:** Spam messages exploit visual urgency through excessive capitalization (`URGENT`, `CLAIM NOW`).
- **Message Length & Word Count:** Legitimate SMS messages tend to be concise, while email phishing templates contain longer narrative lures.

---

### 7. How does the Malicious URL detector work without executing the URL?
**Answer:**
It uses a **static security analysis** approach combining:
1. **Character n-gram TF-IDF (3 to 5 grams):** Analyzes sub-word character sequences, detecting domain typo-squatting, randomized obfuscation, and spoofing strings (e.g. `paypa1`, `sec-upd`).
2. **Lexical Structural Heuristics:** Quantitative features including URL character length, subdomain dot frequency ($>3$), presence of the `@` redirection symbol, raw IP address hostnames (bypassing DNS), absence of HTTPS, and suspicious keyword detection (`login`, `verify`, `wallet`).

---

### 8. What is the difference between ROC-AUC and Precision-Recall AUC (PR-AUC), and which is more informative for imbalanced spam data?
**Answer:**
- **ROC-AUC** plots True Positive Rate vs. False Positive Rate. Because the negative class (Ham) represents ~87% of the data, the False Positive Rate ($\text{FP} / (\text{FP} + \text{TN})$) can remain artificially low even with several false alarms, giving an overly optimistic score.
- **PR-AUC** plots Precision vs. Recall. It focuses directly on the minority class (Spam) without being inflated by a large number of True Negatives. Therefore, in highly imbalanced problems, PR-AUC and $F_1$-score give a much more rigorous assessment of detector reliability.

---

### 9. How did you calculate token attribution / explainability for the end user?
**Answer:**
- For linear models (Linear SVM, Logistic Regression), we extract the learned feature weights (`coef_`) corresponding to the active vocabulary indices of the input message.
- For Naive Bayes, we calculate the log-likelihood ratio: $\log P(w \mid \text{Spam}) - \log P(w \mid \text{Ham})$.
- Tokens with high positive weights are categorized as **Spam Triggers** (e.g. `claim`, `prize`, `call`), while negative weights represent **Ham Indicators** (e.g. `meeting`, `lunch`, `thanks`), allowing the user to understand exactly *why* a message was flagged.

---

### 10. How is model latency addressed for high-throughput production deployment?
**Answer:**
- **Linear Models (Linear SVM & Logistic Regression)** achieve sub-millisecond inference speeds ($\approx 0.0003 - 0.0004\text{ ms}$ per sample), making them capable of handling tens of thousands of requests per second.
- **Caching & Model Pre-loading:** Models are loaded into memory once during the application's startup lifespan rather than re-instantiated on each request.
- **Asynchronous Architecture:** The backend is powered by FastAPI and Uvicorn with ASGI non-blocking event loops and persistent SQLite/MySQL connection pooling.

