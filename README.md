EEG-Based Cognitive State Classification and Learner Attention Recommendation System
Course: Machine Learning Laboratory

Dataset: SEED (Shanghai Jiao Tong University Emotion EEG Dataset)

Approach: Supervised Machine Learning

Language: Python 3.x

1. Abstract
This repository contains an end-to-end machine learning framework developed for the Machine Learning Laboratory. Utilizing the multi-channel SEED dataset, the system is split into two core projects: (1) Emotion Cognitive State Classification into Negative, Neutral, and Positive states, and (2) Leaerner Attention Level Assessment coupled with an automated Video Lecture Recommendation System. By using mathematical optimization for linear models and gradient-boosted trees combined with strict subject-wise cross-validation, we achieve high macro-generalizability and rapid execution times (~20 minutes on a standard CPU) across more than 29,000 data windows.

2. Project Overview
The Problem
Electroencephalogram (EEG) signals provide a rich, continuous window into the human brain's internal states. Automatically decoding these signals into distinct emotional and cognitive profiles unlocks powerful capabilities in adaptive e-learning platforms, passive health monitoring, and closed-loop brain-computer interfaces (BCIs). However, processing multi-channel, high-frequency EEG data at scale presents massive challenges in feature engineering, cross-subject variance, and computational efficiency.

Machine Learning Approach
Subject-wise cross-validation on 15 subjects means only ~12 subjects are available for training at any given time. In such data-constrained regimes, linear models and tree ensembles with smart feature selection are highly appropriate, offering robust mathematical generalization while completely avoiding the computational overhead of deep architectures. This framework favors domain-specific feature extraction paired with robust classical machine learning algorithms. By leveraging optimized models like LinearSVC and Histogram-Based Gradient Boosting (HistGradientBoostingClassifier), we dramatically mitigate overfitting, lower training times by multiple orders of magnitude, and retain architectural interpretability.

The Two Sub-Projects
Project 1: Emotion Classification: Focuses on predicting a user's affective state along three dimensions (Negative, Neutral, and Positive). It validates neural signatures using statistical, spectral, and hemispheric asymmetry features.

Project 2: Attention Level Classification & Video Recommendation: Formulates a novel, continuous Composite Attention Score from established EEG neural markers. It groups windows by lesson trials, classifies student attention into Low, Medium, or High bands, and generates a personalized, ranked recommendation list flagging the top 3 videos where a learner exhibited low engagement and requires material retaking.

3. Dataset Description
The system processes the preprocessed version of the SEED (SJTU Emotion EEG Dataset) provided by Shanghai Jiao Tong University.Dataset Licensing Notice: Due to dataset licensing restrictions, raw .mat files cannot be hosted in this public repository. Users must request and download the official preprocessed dataset directly from the SJTU SEED Official Page.

4. Project 1:
Emotion Classification4.1 ObjectiveTo accurately categorize continuous EEG windows into three emotional archetypes (Negative, Neutral, and Positive) utilizing machine learning classifiers, evaluated under a strict subject-wise $K$-fold cross-validation scheme to completely eliminate inter-subject data leakage.4.2 Feature Extraction Pipeline To transform the multidimensional time-series EEG matrices into discriminative tabular arrays, the following pipeline is executed per subject session:

.mat Files (Pre-computed DE + PSD + DASM + RASM)
                    │
                    ▼
    [Temporal Statistical Extraction]
    Compute: mean, std, skewness, kurtosis across 10-sec time axes
                    │
                    ▼
    [Spectral Band Power Ratios]
    Extract channel-wise cross-band ratios: θ/α, β/α, and γ/β
                    │
                    ▼
    [Hemispheric Asymmetry Indices]
    Extract spatial mean and std profiles across DASM and RASM channels
                    │
                    ▼
    Total Aggregated Feature Space: ~3,516 features per window
                    │
                    ▼
    [Mutual Information Feature Selection]
    SelectKBest (mutual_info_classif) evaluated per fold -> Top 80 features
                    │
                    ▼
    [Downstream Classification]
    6 ML Classifiers processed via Subject-Wise 5-Fold Cross-Validation


4.3 Models Used

All implementations utilize scikit-learn variants optimized for speed and parallel execution:
LinearSVC: Linear Support Vector Classification optimized via SVRG/dual formulations. Replaced standard O(n^3) RBF kernels with an O(n) linear solver to yield a 100x speedup on large arrays.
LDA (Linear Discriminant Analysis): Maximizes the ratio of between-class variance to within-class variance using Singular Value Decomposition (svd solver).
Random Forest: An ensemble of 200 fully grown decision trees utilizing balanced subsampling and square-root feature allocation per node split (n_jobs=-1).
Hist Grad Boost (Histogram-Based Gradient Boosting): An optimized variant of LightGBM built natively within scikit-learn. Discretizes continuous features into 256 integer bins to massively accelerate boosting iterations on 29k+ rows.
KNN (K-Nearest Neighbors): Configured with K=7 using a specialized ball_tree spatial partition algorithm to accelerate cross-distance lookups.
Logistic Reg: Parametrized using the saga solver and multinomial log-loss to handle multi-class optimization via an $L_2$ regularization penalty.

4.4 Results & Visualizations
Feature Selection Optimization Interpretation:
Evaluating mean cross-validation accuracy across a range of feature spaces demonstrates that peak classification stability occurs between 60 and 100 features. Selecting K=80 balances model capacity and performance, successfully paring down the initial ~3,516 feature space without sacrificing signal accuracy.

Comprehensive Classifier Performance Interpretation: LinearSVC dominates the evaluation benchmarks with an overall accuracy of 71.94%. It is closely followed by LDA at 70.11%. Both linear estimators outperform non-linear tree ensembles, confirming that regularized hyperplanes map exceptionally well to high-dimensional EEG spectral spaces.

Error Analysis Matrix Interpretation: Aggregated cross-fold confusion matrices reveal high classification sensitivity within the "Positive" affective state across nearly all models. Misclassifications are predominantly concentrated along the boundaries between adjacent states (e.g., Neutral windows incorrectly categorized as Negative or Positive).

Per-Class Robustness Interpretation: Dissecting F1 scores reveals a consistent trend: every classifier demonstrates optimal performance when isolating Positive states, whereas Neutral states exhibit higher baseline error variance across the board.Cross-Subject Variance 

Profiles Interpretation: The boxplot illustrates distinct variance boundaries across the 5 validation folds. The linear algorithms (LinearSVC, LDA) exhibit a tighter interquartile range (IQR), proving they are less susceptible to individual subject drift than non-linear variants.

Cognitive Frequency Activation Interpretation: Mapping normalized Differential Entropy (DE) across frequency groups confirms deep neural correlation. Positive emotional states show marked elevation in high-frequency Beta and Gamma bands, whereas Negative emotional states correspond to noticeable power suppression in these regions.

Feature Importance Discretization Interpretation: Mutual Information scoring identifies DE_stat and BandRatio variables as the primary drivers of class discriminability. Cross-band combinations and differential entropy parameters prove 3.5x more informative than raw power spectral readings.

Inter-Subject Generalizability (LOSO Framework) Interpretation: Running a rigorous Leave-One-Subject-Out (LOSO) evaluation using LinearSVC shows that while mean baseline performance hovers comfortably at 70.2%, individual subject accuracies range from a low of 52% (Subject 4) to a high of 88% (Subject 12). This highlights the extensive biological variance native to human EEG data.

Demographic Divergence Interpretation: Isolating LOSO outcomes across demographic cohorts reveals stable classification performance across male and female sub-groups, demonstrating that the structural feature sets remain generalizable across genders.

4.5 Key Findings
Linear Dominance: LinearSVC achieves a top performance of 71.94% accuracy under rigorous subject-independent evaluation conditions, demonstrating excellent mathematical generalizability.

Spectral Powerhouse: Multi-band ratio metrics (such as Beta/Alpha and Theta/Alpha combinations) exhibit significantly higher mutual information scores compared to raw time-domain statistical descriptors.

Affective Signatures: Positive emotional states generate distinct, easily classified cortical activations characterized by sharp increases in high-frequency Beta and Gamma oscillations.


5. Project 2: Attention Classification & Recommendation
5.1 Objective:To develop a passive cognitive model capable of assessing a learner's instantaneous attention depth (Low, Medium, or High) during educational video viewing, and to build an intelligent recommender system that automatically flags and ranks specific video lectures that a user failed to engage with and should retake.

5.2 Attention Label Derivation: Because the SEED dataset contains native labels for emotional valence rather than cognitive workload, an empirical proxy model was constructed to derive continuous attention baselines. For every 10-second sliding window, a Composite Attention Score is computed using three validated neuro-cognitive markers: Theta/Beta Ratio: Broadly established in clinical neurofeedback; an increase in slow-wave Theta power relative to fast-wave Beta power reflects a state of inattention or daydreaming.
Frontal Relative Beta Suppression: Tracks relative Beta power across frontal channels (Fp1, Fp2, F3, F4, F7, F8, Fz). Higher localized Beta indicates active executive function and working memory engagement.DASM Beta Asymmetry ($A_{DASM}$): Computes left-to-right hemispheric imbalances in high-frequency bands; positive asymmetry scores correlate with approach motivation and active attention.The formal mathematical formulation is defined as.

Valuable Structural Scaling: This composite value is structured such that a LOWER score indicates higher cognitive attention (high beta, low theta, strong frontal engagement). Windows are transformed into discrete target labels using strict per-subject percentiles:High Attention (Class 2): Bottom 33% of raw scores.Medium Attention (Class 1): Middle 33% of raw scores.Low Attention (Class 0): Top 33% of raw scores.

5.3 Recommendation Engine Architecture
                        Per-Window Raw EEG Signal
                                    │
                                    ▼
                [Feature Extraction & Scaler Pipeline]
                                    │
                                    ▼
            [Best Trained Attention Classifier (HistGradBoost)]
            Predict discrete attention label per 10-second window
                                    │
                                    ▼
                [Subject & Trial Window Aggregation]
        Group all window predictions by unique video trial ID blocks
                                    │
                                    ▼
                [Continuous Mean Predicted Label Index]
    Compute mean attention level score per video block (Continuous Scale [0, 2])
                                    │
                                    ▼
                    [Recommender Matrix Inversion]
    Sort video items in ascending order (lowest attention scores ranked first)
                                    │
                                    ▼
    Final Output: Flag Top 3 Low-Attention items for review/retaking

5.4 Results & Visualizations 
Attention Feature Optimizations Interpretation: Feature mapping for the derived attention space indicates an optimal selection threshold at $K=60$ components. Beyond this point, tracking further features introduces semantic noise that degrades model accuracy.

Attention Classification Profiles
Interpretation: Within the synthetic attention space, HistGradientBoostingClassifier demonstrates superior multi-class parsing capabilities, delivering a cross-validation accuracy of 57.73%. It outpaces traditional linear models by capturing complex non-linear feature interactions across the multi-channel grid.

Attention Error Profiling Interpretation: The multi-class attention matrix shows that classification errors occur almost exclusively between adjacent categories . Crucially, the model rarely confuses polar opposites, confirming the validity of the underlying scoring boundaries.

Class Stabilization Plots Interpretation: In tree-based ensembles, F1 scores remain uniform across all target classes. This uniformity verifies that our percentile-based label splitting successfully mitigates class imbalance issues.

Boosting Run Consistency Interpretation: HistGradientBoostingClassifier exhibits exceptionally low accuracy variance across all validation folds. This minimal spread confirms that gradient boosting architectures are highly stable when handling derived cognitive indexes.

Spectral Attention Profiles Interpretation: Validating our derived labels against raw feature inputs confirms a strong neurobiological alignment: windows assigned to the High Attention class exhibit pronounced elevation in active Beta and Gamma bands, whereas Low Attention windows are characterized by dominant slow-wave Theta power. 

Score Threshold Validation Interpretation: The probability density functions (PDF) and corresponding boxplots demonstrate clean, mathematical separation across the derived class thresholds. This visual alignment confirms that our percentile-based binning strategy operates reliably across the dataset.

The Personalized Recommendation Matrix Interpretation: This heatmap displays the final recommendation matrix (Subjects vs. Video Trials). Bright red cells indicate video blocks where a specific subject's attention plummeted into the lowest tier. The system flags these regions, dynamically generating a targeted intervention list that prompts the user to rewatch those specific lectures.

System Performance Benchmarks Interpretation: Evaluating the system's top-3 recommendations demonstrates a Precision@3 of 0.706 and a Recall@3 of 0.612. These metrics significantly outperform a random baseline recommendation engine ($0.333$). This leap in performance confirms that the model accurately isolates and targets true periods of cognitive disengagement.

Global User Profiles Interpretation: Tracking overall predicted attention levels across the participant pool reveals clear individual variations. Users such as Subject 4 maintain consistently high attention profiles throughout the sessions, whereas Subject 0 exhibit lower aggregate engagement scores and require more frequent content reviews.

6. Limitations & Ethical Considerations
Sample Scale Constraints: While the SEED dataset provides high channel density and excellent spatial resolution, it contains data from only 15 unique participants. Consequently, the high cross-validation accuracies reported here may fluctuate when deploying the models to larger, more ethnically and behaviorally diverse populations.

Synthetic Ground-Truth Approximations: Because the underlying dataset lacks direct, self-reported attention metrics, the attention labels were constructed using a proxy neurofeedback scoring formula. Future iterations of this framework will incorporate explicit gaze-tracking telemetry and synchronized performance assessments to empirically validate these cognitive states.

Privacy and Cognitive Monitoring Boundaries: Continuous EEG analysis and passive cognitive modeling tracking engagement patterns present clear ethical challenges. Systems of this nature must implement robust data encryption pipelines and maintain strict user consent protocols to prevent unauthorized behavioral tracking or employee surveillance.
