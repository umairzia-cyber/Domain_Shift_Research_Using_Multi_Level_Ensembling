# Domain_Shift_Research_Using_Multi_Level_Ensembling

This project implements a multi-level hierarchical stacking framework for classification under domain shift, with optional synthetic data augmentation and feature alignment.

It compares three stacked learning architectures—Original, Residual, and Entropy-aware pipelines—each built using multi-stage ensembles (Logistic Regression, Random Forest, Extra Trees, and XGBoost). Out-of-fold (OOF) predictions are used to prevent leakage and construct meta-features across three hierarchical levels (E0 → E1 → E2).

To improve robustness, the pipeline includes:

CTGAN-based synthetic data generation to address class imbalance
CORAL (Correlation Alignment) to reduce domain shift by aligning feature distributions between source and target domains
Entropy-based uncertainty features to enrich higher-level models with confidence information

Evaluation is performed using ROC-AUC, F1-score, accuracy, and classification reports, with threshold tuning on F1-score for optimal decision boundaries.

Experiments are run across:

Multiple synthetic data ratios
In-domain vs domain-shifted settings
CORAL-aligned feature space

Overall, the framework provides a systematic comparison of hierarchical ensemble strategies for robust classification under dataset shift conditions.
