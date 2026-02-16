# multimodal-emotion-recognition

Multimodal Emotion Recognition System
Overview

This project implements a multimodal emotion recognition pipeline using:

Speech-only modelling (acoustic features)

Text-only modelling (contextual transformer)

Multimodal fusion (feature-level concatenation)

The dataset used is the Toronto Emotional Speech Set.

Dataset

TESS contains emotionally spoken single-word utterances across 7 emotions:

Angry

Disgust

Fear

Happy

Neutral

Pleasant Surprise (ps)

Sad

Important note:

The spoken words are semantically neutral (e.g., “chair”, “dog”), meaning that emotion is primarily encoded in prosody rather than lexical content.

Architecture
1️⃣ Speech-Only Model (Temporal Modelling Block)
Pipeline

Audio → MFCC extraction → Mean pooling → MLP classifier

Details

40-dimensional MFCC features (using librosa)

Temporal aggregation via mean pooling

Fully connected neural network

CrossEntropyLoss

Adam optimizer

Rationale

MFCCs effectively capture spectral and prosodic cues relevant to emotion classification. Given the dataset size, a lightweight MLP was chosen to avoid overfitting.

2️⃣ Text-Only Model (Contextual Modelling Block)
Architecture

Fine-tuned BERT-base-uncased with classification head.

Rationale

BERT captures contextual semantic representations. However, since TESS contains emotionally neutral words, the text modality lacks discriminative emotional information.

Result:

Text-only accuracy ≈ random baseline (~0.15 for 7 classes).

3️⃣ Multimodal Fusion Model (Fusion Block)
Architecture

Speech branch → Linear projection

Text branch → Linear projection

Feature concatenation

Final MLP classifier

Fusion type: Late fusion at representation level.

Rationale

Late fusion allows independent modality learning and prevents noisy text features from degrading performance.

Experiments
Speech-Only

20 epochs

Adam optimizer

Strong convergence

High test accuracy

Text-Only

5 epochs

Fine-tuned BERT

Accuracy ≈ 0.15 (near random baseline)

Multimodal Fusion

Feature concatenation

Accuracy comparable to speech-only model

Observation:

Fusion does not significantly outperform speech-only because text modality lacks complementary emotional signal in TESS.

Analysis
Easiest Emotions

Angry

Happy

Reason:
High pitch variation and strong energy patterns.

Hardest Emotions

Neutral

Sad

Pleasant Surprise

Reason:
Subtle prosodic differences and overlapping spectral characteristics.

When Does Fusion Help?

Fusion is most beneficial when modalities provide complementary signals.

In TESS:

Speech dominates

Text adds minimal information

For stronger multimodal benefit, datasets such as IEMOCAP would be more suitable.
