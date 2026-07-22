# Ransomware Classification Using Multimodality

A deep learning-based multimodal framework for ransomware detection and classification by integrating **static analysis, dynamic behavior analysis, and network traffic analysis**. The project combines multiple security perspectives to improve ransomware identification accuracy and robustness.

## 📌 Overview

Ransomware continues to evolve rapidly, making traditional signature-based detection methods insufficient against new and unknown variants. This project proposes a **multimodal learning approach** that combines heterogeneous malware features to classify ransomware families and distinguish malicious samples from benign files.

The framework integrates:

- **Static Analysis** – Extracts characteristics from executable files without execution.
- **Dynamic Analysis** – Captures behavioral patterns during malware execution.
- **Network Analysis** – Analyzes communication patterns and network-based indicators.

The extracted features are learned using deep learning models and fused to achieve improved ransomware classification performance.

---

## 🏗️ Project Architecture

The framework follows a multi-agent pipeline:

```
                  Malware Samples
                         |
        -----------------------------------
        |                |                |
 Static Analysis   Dynamic Analysis   Network Analysis
        |                |                |
        -----------------------------------
                         |
                 Feature Fusion Module
                         |
              Ransomware Classification
                         |
              Prediction & Evaluation
```

---

## 🔍 Key Components

### 1. Static Analysis Agent
Analyzes executable characteristics without running the malware.

Features include:
- PE file characteristics
- Opcode/instruction patterns
- File metadata
- Static behavioral indicators


### 2. Dynamic Analysis Agent
Captures runtime behavior of ransomware samples.

Analyzed features include:
- API calls
- File operations
- Process behavior
- Registry modifications
- System activities


### 3. Network Analysis Agent
Extracts network-level indicators.

Features include:
- Network flows
- Communication patterns
- Traffic statistics
- Connection behavior


### 4. Fusion & Classification Module

Combines multimodal features using deep learning techniques to improve classification performance.

---

## 🧠 Machine Learning Approach

The project uses:

- Deep Learning Models
- Multimodal Feature Fusion
- Representation Learning
- Contrastive Learning
- Supervised Classification

### Deep Contrastive Autoencoder (DCAE)

The static feature representation is learned using a Deep Contrastive Autoencoder architecture:

```
Input Features
      |
1901 Dimensions
      |
1024
      |
512
      |
256
      |
128 Latent Representation
```

The learned embeddings are combined with dynamic and network representations for final classification.

---

## 📂 Repository Structure

```
Ransomware-Classification-Using-Multimodality
│
├── agents/
│   ├── Static Analysis Agent
│   ├── Dynamic Analysis Agent
│   ├── Network Analysis Agent
│   └── Fusion Agent
│
├── models/
│   ├── Deep Learning Models
│   └── Classification Models
│
├── utils/
│   ├── Data Processing
│   ├── Feature Extraction
│   └── Helper Functions
│
├── run_new.py
│   └── Main execution pipeline
│
├── zerday.py
│   └── Zero-day detection module
│
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/AimenWadood/Ransomware-Classification-Using-Multimodality.git
```

Navigate to the project directory:

```bash
cd Ransomware-Classification-Using-Multimodality
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the complete classification pipeline:

```bash
python run_new.py
```

The system will:

1. Load extracted malware features
2. Process multimodal representations
3. Perform feature fusion
4. Classify ransomware samples
5. Generate evaluation results

---

## 📊 Evaluation Metrics

The model performance can be evaluated using:

- Accuracy
- Macro F1-score
- Precision
- Recall
- ROC-AUC
- Expected Calibration Error (ECE)
- Negative Log Likelihood (NLL)

---

## 🛡️ Applications

This framework can support:

- Malware detection systems
- Endpoint security solutions
- SOC threat detection pipelines
- Automated malware analysis
- Ransomware family classification

---

## 🔬 Research Contribution

This project demonstrates how combining multiple security modalities can improve ransomware detection compared to single-source analysis approaches.

Key contributions:

- Multimodal ransomware classification framework
- Integration of static, dynamic, and network intelligence
- Deep representation learning for malware features
- Improved detection of unknown ransomware behavior

---

## 👩‍💻 Author

**Aimen Wadood**  
MS Cybersecurity Researcher  
GitHub: https://github.com/AimenWadood

---

## 📄 License

This project is licensed under the GPL License.
