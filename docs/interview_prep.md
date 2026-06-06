# Interview Preparation Guide
## AI-Powered Medical Image Disease Detection Platform

---

# SECTION 1 — CNN & DEEP LEARNING QUESTIONS

---

**Q1: Explain the architecture of your custom CNN model.**

**Answer:**
My custom CNN follows a classic VGG-inspired design with 4 convolutional blocks followed by a global average pooling layer and a dense classification head.

Each block contains:
- `Conv2D` — learns spatial feature detectors
- `BatchNormalization` — stabilizes training by normalizing activations
- `ReLU activation` — introduces non-linearity, prevents vanishing gradients
- `MaxPooling2D` — reduces spatial dimensions, provides translation invariance

The channel progression (32 → 64 → 128 → 256) follows the standard doubling pattern: as spatial size halves with each MaxPool, we double channels to maintain representational capacity.

The `GlobalAveragePooling2D` replaces Flatten to reduce parameters and overfitting, and adds spatial regularization. The final head is `Dense(256) → Dropout(0.5) → Dense(1, sigmoid)`.

**Follow-up: Why GlobalAveragePooling instead of Flatten?**
Flatten outputs `7×7×256 = 12,544` features. GlobalAveragePooling outputs 256. This reduces overfitting, is more parameter-efficient, and acts as a structural regularizer as described in the Network-in-Network paper.

---

**Q2: What is the role of BatchNormalization?**

**Answer:**
BatchNormalization normalizes each mini-batch's activations to have zero mean and unit variance, then applies learnable scale (γ) and shift (β) parameters.

Benefits:
1. Reduces internal covariate shift
2. Allows higher learning rates → faster training
3. Acts as a regularizer (reduces need for Dropout)
4. Makes the model less sensitive to weight initialization

During inference, uses running mean/variance computed during training, not the current batch statistics.

**Follow-up: Where do you place BatchNorm — before or after activation?**
Theoretically after Conv and before activation (Conv → BN → ReLU). In practice, the original paper placed it after activation. Modern practitioners often use Conv → BN → ReLU. In my code, I follow the Conv → BN → ReLU ordering.

---

**Q3: Why is Dropout used and how does it work?**

**Answer:**
Dropout randomly sets a fraction of neurons to zero during each forward pass in training. With rate=0.5, each neuron has a 50% probability of being dropped.

Intuition: This forces the network to learn redundant representations. No single neuron can rely on any other neuron being present, so each learns more robust features.

During inference, Dropout is disabled and all neurons are active, but their outputs are scaled by (1 - dropout_rate) to maintain the expected activation magnitude.

In my model, I use `Dropout(0.5)` in the dense head to prevent overfitting to the training set.

---

**Q4: Explain vanishing and exploding gradients. How does your architecture address them?**

**Answer:**
**Vanishing gradients:** In deep networks, backpropagation multiplies many small gradient values together. With sigmoid activations and deep networks, gradients can become exponentially small, making early layers learn very slowly.

**Exploding gradients:** The opposite — gradients grow exponentially, causing unstable training (NaN losses).

My architecture addresses these:
- **ReLU activations** — gradient is either 0 or 1 (no squashing like sigmoid)
- **BatchNormalization** — keeps activations in a reasonable range
- **He initialization** — sets initial weights based on fan-in, appropriate for ReLU
- **Gradient clipping** (in optimizer config) — prevents explosions

---

**Q5: What is the difference between underfitting and overfitting? How did you detect and prevent them?**

**Answer:**
**Underfitting:** Model is too simple to capture the patterns in data. Symptoms: low training accuracy AND low validation accuracy.

**Overfitting:** Model memorizes training data but doesn't generalize. Symptoms: high training accuracy, significantly lower validation accuracy.

**Detection in my project:**
- I monitor `val_recall` and `train_recall` during training using TensorBoard
- A large gap (> 5%) indicates overfitting

**Prevention in my project:**
- `Dropout(0.5)` in the dense head
- `L2 regularization` (kernel_regularizer) on Conv layers
- `EarlyStopping` — stops training when val_recall stops improving
- `Data augmentation` — artificially increases training diversity
- `GlobalAveragePooling` — fewer parameters vs Flatten

---

# SECTION 2 — TRANSFER LEARNING QUESTIONS

---

**Q6: What is transfer learning and why did you use it?**

**Answer:**
Transfer learning takes a model pre-trained on a large dataset (ImageNet — 1.4M images, 1000 classes) and adapts it to a new task (pneumonia detection — 5,216 images, 2 classes).

**Why it works:**
Deep neural networks learn hierarchical features:
- Early layers: edges, gradients (universal, task-independent)
- Middle layers: textures, patterns (moderately transferable)
- Later layers: high-level semantics (task-specific)

We reuse the transferable early and middle layers and replace only the task-specific head.

**Benefits for my project:**
1. Much less data needed (5,216 images instead of millions)
2. Faster convergence (starts from good features, not random)
3. Better generalization (rich ImageNet features)

---

**Q7: Explain your two-phase transfer learning strategy.**

**Answer:**
**Phase 1 — Head Warm-up (5 epochs, LR=0.001):**
- Freeze ALL base model layers (`base_model.trainable = False`)
- Train only the custom classification head
- Goal: Initialize the new head to reasonable values without corrupting the pretrained weights
- Problem if skipped: Large random gradients from the untrained head would destroy pretrained weights

**Phase 2 — Fine-tuning (45 epochs, LR=0.0001):**
- Unfreeze the top 30% of base model layers
- Train both the unfrozen base layers AND the head
- Learning rate is 10× smaller to make gentle updates
- Goal: Adapt the high-level features to X-ray domain features

**Follow-up: Why 10× smaller learning rate in Phase 2?**
The pretrained weights encode valuable patterns. Large updates would overwrite this knowledge. We want to nudge them toward the new domain, not reinitialize them randomly.

---

**Q8: Compare ResNet50, DenseNet121, and EfficientNetB0.**

**Answer:**

**ResNet50 (He et al., 2015):**
- 50 layers with residual (skip) connections: `output = F(x) + x`
- Skip connections solve the degradation problem in very deep networks
- ~25M parameters
- Strong baseline, well-tested

**DenseNet121 (Huang et al., 2016):**
- Each layer connects to ALL subsequent layers: dense skip connections
- Feature reuse reduces parameter count significantly
- ~8M parameters, very memory efficient
- Originally developed FOR medical imaging (CheXNet paper used DenseNet)
- Often best performer for chest X-ray tasks

**EfficientNetB0 (Tan & Le, 2019):**
- Compound scaling of width/depth/resolution simultaneously
- ~5.3M parameters — most efficient
- Best accuracy/parameter ratio

**In my project:** DenseNet121 consistently achieves highest recall for pneumonia detection, aligning with the CheXNet research.

---

# SECTION 3 — MEDICAL AI & EVALUATION QUESTIONS

---

**Q9: Why is Recall more important than Accuracy in medical diagnosis?**

**Answer:**
Consider a test set with 70% pneumonia cases and 30% normal cases.

A "stupid" classifier that predicts PNEUMONIA for everything would achieve:
- Accuracy: 70% (looks decent!)
- Recall: 100% (catches every pneumonia case)
- Precision: 70% (70% of pneumonia predictions are correct)
- BUT it flags every healthy person as sick

The critical failure mode for medical AI:
- **False Negative (FN):** Patient HAS pneumonia, AI says NORMAL → patient goes home untreated → potentially life-threatening
- **False Positive (FP):** Patient is NORMAL, AI says pneumonia → additional tests, cost, anxiety

FN >> FP in consequence → **we must maximize Recall (Sensitivity)**.

In my model, I:
1. Set recall as the primary optimization metric in callbacks
2. Weight class_weight toward PNEUMONIA class
3. Tuned decision threshold (default 0.5) to shift toward higher recall

**Follow-up: What is the Precision-Recall tradeoff?**
Lowering the classification threshold (e.g., from 0.5 to 0.3) increases recall (more positives predicted) but decreases precision (more false positives). The ROC curve and PR curve visualize this tradeoff.

---

**Q10: What is ROC-AUC and why did you report it?**

**Answer:**
**ROC** = Receiver Operating Characteristic curve — plots True Positive Rate (Recall) vs False Positive Rate at every possible classification threshold.

**AUC** = Area Under the Curve = probability that the model ranks a random positive sample higher than a random negative sample.

- AUC = 0.5 → random classifier
- AUC = 1.0 → perfect classifier
- AUC = 0.97 (my DenseNet121) → 97% chance of correctly ranking a pneumonia case above a normal case

**Advantages of AUC:**
1. Threshold-independent — evaluates the model at ALL thresholds
2. Useful for imbalanced datasets
3. Directly interpretable as a probability

---

**Q11: Explain class imbalance and how you handled it.**

**Answer:**
My training set has 3,875 PNEUMONIA and 1,341 NORMAL images — a 2.9:1 ratio.

If not addressed, the model learns to predict PNEUMONIA more often simply because it's more common, not because it learned anything meaningful.

**My solutions:**

1. **Class weights:** Computed as `{0: 2.89, 1: 1.0}` — each NORMAL sample contributes 2.89× more to the loss, making them equally important despite being fewer. Applied via `class_weight` parameter in `model.fit()`.

2. **Data augmentation:** Applied only to training set. Random flips, rotations, brightness changes artificially increase diversity.

3. **Evaluation metrics:** Used recall, F1, ROC-AUC — not accuracy. Accuracy is misleading with class imbalance.

4. **Threshold tuning:** After training, I evaluated different thresholds (0.3 to 0.7) on the validation set and chose the threshold maximizing recall while keeping precision above 80%.

---

# SECTION 4 — SQL & DATABASE QUESTIONS

---

**Q12: Explain your database schema and design decisions.**

**Answer:**
I designed a 5-table schema following medical records standards:

**patients** — Master patient registry
- Soft delete (`is_active` flag) — never hard delete medical records
- `created_at` timestamp for audit trail

**predictions** — Every AI inference result
- Foreign key to `patients` (many predictions per patient)
- Stores both prediction label AND raw probabilities (for threshold adjustment without rerunning the model)
- `processing_time_ms` for performance monitoring

**image_metadata** — Technical image details
- Foreign key to `predictions`
- Stores image hash (SHA-256) for duplicate detection
- Stores original dimensions for quality control

**model_versions** — Model registry
- Stores all performance metrics at registration time
- `is_active` flag to mark the currently deployed model

**audit_logs** — Compliance and security
- Immutable record of all database actions
- Critical for HIPAA compliance in real systems

**Follow-up: Why SQLite as default? When would you use PostgreSQL?**
SQLite is serverless — zero setup, single file. Perfect for development, demos, and single-instance deployments. PostgreSQL for production: multi-user access, concurrent writes, better performance at scale, full ACID compliance, and advanced features like row-level security.

---

**Q13: Write a SQL query to find the monthly pneumonia detection rate.**

**Answer:**
```sql
SELECT 
    strftime('%Y-%m', pr.created_at)              AS month,
    COUNT(*)                                        AS total_predictions,
    SUM(
        CASE WHEN pr.prediction_label = 'PNEUMONIA' 
             THEN 1 ELSE 0 END
    )                                               AS pneumonia_cases,
    ROUND(
        100.0 * SUM(
            CASE WHEN pr.prediction_label = 'PNEUMONIA' 
                 THEN 1 ELSE 0 END
        ) / COUNT(*),
        2
    )                                               AS pneumonia_rate_pct
FROM      predictions   pr
JOIN      patients       pa ON pr.patient_id = pa.id
WHERE     pa.is_active = 1
  AND     pr.created_at >= date('now', '-6 months')
GROUP BY  month
ORDER BY  month;
```

---

**Q14: What is the difference between WHERE and HAVING clauses?**

**Answer:**
- `WHERE` filters **individual rows** before any aggregation
- `HAVING` filters **groups** after `GROUP BY` aggregation

Example: Finding hospitals with pneumonia rate > 50%
```sql
SELECT 
    pa.hospital,
    COUNT(*) AS total,
    100.0 * SUM(CASE WHEN prediction_label='PNEUMONIA' THEN 1 ELSE 0 END) / COUNT(*) AS pneu_rate
FROM predictions pr
JOIN patients pa ON pr.patient_id = pa.id
WHERE pr.created_at >= '2024-01-01'      -- WHERE: filter rows first
GROUP BY pa.hospital
HAVING pneu_rate > 50.0                   -- HAVING: filter groups after aggregation
ORDER BY pneu_rate DESC;
```

---

# SECTION 5 — MLOPS QUESTIONS

---

**Q15: What is MLOps and how have you implemented it?**

**Answer:**
MLOps (Machine Learning Operations) applies DevOps principles to the machine learning lifecycle. It addresses the gap between model development and production deployment.

**My implementations:**

1. **Experiment Tracking (MLflow):**
   - Every training run logs: hyperparameters, metrics, model artifacts
   - Easy comparison of runs: which LR/batch size/architecture performed best?
   - Reproducibility: given a run_id, anyone can reproduce the exact experiment

2. **Model Versioning:**
   - Dual tracking: SQLAlchemy DB + MLflow Registry
   - Each registered model has: name, version, metrics, path, timestamp
   - `is_active` flag marks the current production model

3. **Prediction Tracking:**
   - Every inference saved to database with timestamp, confidence, model version
   - Enables performance monitoring over time

4. **Drift Detection:**
   - I compare the mean confidence score of recent predictions (last 100) against the baseline (training period average)
   - Significant drop in confidence may indicate distribution shift (new type of X-ray machine, different patient population)

5. **Logging:**
   - Structured logging with colorlog
   - Rotating file handlers (prevents disk fill)
   - Log levels: DEBUG, INFO, WARNING, ERROR

---

**Q16: What is model drift and how do you detect it?**

**Answer:**
**Concept drift:** The statistical relationship between input X and output Y changes over time. For chest X-rays: a new type of X-ray machine produces different image characteristics, causing model performance to degrade.

**Data drift:** The distribution of inputs (X) changes. Example: hospital starts imaging a different demographic.

**Detection methods I implemented:**
1. **Mean confidence monitoring:** If average confidence drops significantly, the model is less certain — possible drift
2. **Statistical tests:** Kolmogorov-Smirnov test comparing recent confidence distribution to baseline
3. **Performance tracking:** If labeled data is available, track accuracy/recall over time

**Response to detected drift:**
1. Collect samples from the new distribution
2. Fine-tune the model on new data
3. Re-evaluate and re-register if metrics improve

---

# SECTION 6 — STREAMLIT & WEB APP QUESTIONS

---

**Q17: How does Streamlit work and what are its advantages?**

**Answer:**
Streamlit is a Python web framework that turns data science scripts into interactive web apps with minimal code.

**How it works:**
- Every time the user interacts with a widget, the entire Python script re-runs from top to bottom
- `st.session_state` persists values across re-runs
- Streamlit handles all the HTML/CSS/JavaScript — you write only Python

**Advantages:**
1. Zero web development knowledge needed (no HTML/CSS/JS)
2. Rapid prototyping (hours vs days for Flask/Django)
3. Native support for data science libs (Pandas, Plotly, Matplotlib)
4. Easy deployment (Streamlit Community Cloud)

**Limitations:**
1. Not suited for high-traffic production apps (re-run model on every widget change)
2. No multi-user session isolation by default
3. Limited customization compared to React/Vue

**Follow-up: How do you handle state in Streamlit?**
`st.session_state` is a dictionary-like object that persists across re-runs for a single user session. I use it to store: current patient form values, last prediction result, prediction history, and model loading state.

---

**Q18: How does your prediction page work end-to-end?**

**Answer:**
1. User uploads JPEG image via `st.file_uploader`
2. `validate_image_upload()` checks: file type (JPEG/PNG), file size (< 10 MB)
3. Image is decoded using `cv2.imdecode()` from byte array
4. `preprocess_for_model()` applies: grayscale → CLAHE → denoise → resize(224×224) → normalize → expand_dims
5. Model loaded from `models/saved_models/` (cached with `@st.cache_resource`)
6. `model.predict()` returns raw probability (0–1)
7. `format_prediction_result()` converts probability to: label, confidence, risk_level, recommendation
8. Results displayed: confidence gauge (Plotly), probability bar, recommendation text
9. On "Save to Database": patient record inserted + prediction record linked via foreign key
10. `write_audit_log()` records the action

---

# SECTION 7 — COMPUTER VISION QUESTIONS

---

**Q19: Explain CLAHE and why you used it for X-rays.**

**Answer:**
**CLAHE = Contrast Limited Adaptive Histogram Equalization**

Standard histogram equalization distributes intensity globally. For chest X-rays, this can over-amplify noise in already-bright regions while not helping the darker regions enough.

CLAHE divides the image into small tiles (default 8×8) and applies histogram equalization independently in each tile, then uses bilinear interpolation at tile boundaries to prevent artifacts.

The "contrast limited" part caps the histogram amplification at `clipLimit` to prevent noise amplification.

**Why for X-rays:**
- Chest X-rays have highly variable exposure levels (under/over-exposed)
- Pneumonia infiltrates can be subtle in low-contrast regions
- CLAHE enhances local contrast in the areas where infiltrates appear
- Helps the CNN detect subtle features in both bright (bones) and dark (lung fields) regions

---

**Q20: What image augmentations did you use and why?**

**Answer:**
Augmentations artificially expand the training set and teach the model invariance to irrelevant transformations.

| Augmentation | Range | Reasoning |
|---|---|---|
| Horizontal flip | Random | Lungs are roughly symmetric; both orientations are valid |
| Rotation | ±15° | Patient positioning varies; mild rotation should not affect diagnosis |
| Width shift | ±10% | Patient not always perfectly centered |
| Height shift | ±10% | Same as above |
| Brightness | 0.7–1.3× | X-ray exposure varies between machines/technicians |
| Zoom | ±10% | Magnification varies |

**Augmentations I intentionally excluded:**
- Vertical flip: Upside-down lungs don't occur clinically
- Shear: Distorts anatomy in a way that doesn't occur in real images
- Contrast: I apply CLAHE as preprocessing; additional augmentation could confuse it

**Key rule:** Never augment the test/validation set — only training data.

---

# SECTION 8 — SYSTEM DESIGN QUESTIONS

---

**Q21: How would you scale this system to handle 10,000 predictions per day?**

**Answer:**
The current architecture (SQLite + Streamlit single-server) would not handle that load. Here's my scaling plan:

**Backend:**
- Replace Streamlit with FastAPI (REST API) for predictions
- Load model once at startup, not per request
- Use async inference or a prediction queue (Celery + Redis)
- Containerize with Docker, orchestrate with Kubernetes

**Database:**
- Replace SQLite with PostgreSQL (AWS RDS / Azure Database)
- Add read replicas for analytics queries
- Partition the predictions table by month

**Model Serving:**
- TensorFlow Serving or NVIDIA Triton Inference Server
- Model loaded in GPU memory, handles concurrent requests
- Auto-scaling based on request queue depth

**Caching:**
- Cache model predictions for duplicate images (content hash lookup)
- Cache analytics queries (refresh every 5 minutes, not per request)

**Monitoring:**
- Prometheus + Grafana for infrastructure metrics
- MLflow for model performance metrics
- PagerDuty alerts for prediction failure rates

---

**Q22: What security considerations are important for a medical AI system?**

**Answer:**
1. **Authentication & Authorization:** Every request authenticated (JWT tokens), role-based access (radiologist vs admin vs patient)

2. **Data encryption:** X-ray images and patient data encrypted at rest (AES-256) and in transit (TLS 1.3)

3. **Audit logging:** All data access logged (user, timestamp, action) — I've implemented `audit_logs` table

4. **De-identification:** Patient data anonymized in analytics/reporting (HIPAA requirement)

5. **Input validation:** Sanitize all uploaded files (I check file type, size, decode before processing)

6. **API rate limiting:** Prevent abuse (100 predictions/hour per user)

7. **Model explainability:** Grad-CAM visualizations to show WHY the model made a prediction (important for clinician trust and liability)

8. **Compliance:** HIPAA (US), GDPR (EU), ISO 13485 (medical devices)

---

# SECTION 9 — POWER BI QUESTIONS

---

**Q23: What DAX measures did you create and why?**

**Answer:**
Key measures I implemented:

```dax
-- Why DIVIDE() instead of / operator?
-- DIVIDE() handles division by zero gracefully (returns 0 or custom value)
Pneumonia Rate = DIVIDE([Pneumonia Cases], [Total Predictions], 0)

-- Why Rolling 3M Average?
-- Smooths out weekly spikes to show underlying trend
Rolling 3M = AVERAGEX(
    DATESINPERIOD(predictions[created_at], LASTDATE(predictions[created_at]), -3, MONTH),
    [Total Predictions]
)
```

**Design principles:**
- Prefer measures over calculated columns (measures are calculated on demand, columns stored)
- Use CALCULATE() to modify filter context
- Time intelligence functions (PREVIOUSMONTH, DATESINPERIOD) for trend analysis

---

**Q24: Explain the difference between a Calculated Column and a Measure in Power BI.**

**Answer:**

| | Calculated Column | Measure |
|---|---|---|
| Evaluated | At data refresh | At query time |
| Storage | Stored in model | Computed on demand |
| Context | Row context | Filter/aggregation context |
| Memory | Uses more RAM | More memory-efficient |
| Use case | Category/bucket labels | KPIs, aggregations |

**Example:**
- `AgeGroup` — Calculated Column (categorizes each patient row once)
- `Pneumonia Rate` — Measure (recalculates based on current slicer/filter context)

**Rule of thumb:** If you need to use it in a filter or slicer, use a calculated column. If you need an aggregation displayed in a visual, use a measure.

---

# QUICK REFERENCE — TECHNICAL TERMS

| Term | Definition |
|---|---|
| Epoch | One complete pass through the entire training dataset |
| Batch | Subset of training data processed before weight update |
| Learning rate | Step size for gradient descent parameter updates |
| Overfitting | Model performs well on training data but poorly on new data |
| Regularization | Techniques to reduce overfitting (L2, Dropout, BatchNorm) |
| Fine-tuning | Training a pre-trained model on new data with small LR |
| Softmax | Outputs probabilities summing to 1 (multiclass) |
| Sigmoid | Outputs probability for binary classification |
| ROC-AUC | Threshold-independent classification performance metric |
| Recall/Sensitivity | TP / (TP + FN) — fraction of actual positives correctly identified |
| Precision | TP / (TP + FP) — fraction of predicted positives that are actually positive |
| F1 Score | Harmonic mean of Precision and Recall |
| Confusion Matrix | Table showing TP, TN, FP, FN |
| Class weights | Compensation for class imbalance in loss calculation |
| MLflow | Open-source platform for ML lifecycle management |
| CLAHE | Contrast-limited adaptive histogram equalization |
| Transfer learning | Using a pre-trained model's weights for a new task |

---

*Good luck in your interviews! The key is to explain your design decisions, not just describe what you built.*
