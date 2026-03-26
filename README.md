# Multimodal Fake Job Postings Detection

A comprehensive machine learning project that combines textual and visual features to detect fraudulent job postings using multimodal analysis. This project enhances traditional text-based fraud detection by incorporating company profile images scraped from search results.

## Overview

This project addresses the growing problem of fake job postings on online platforms by developing a multimodal detection system that analyzes both textual content and visual elements (company profile images) to improve fraud detection accuracy.

The original dataset contained only textual features from job postings. To enable multimodal analysis, we developed an automated image collection system that scrapes company profile images from web search results, significantly enhancing the dataset's capabilities for comprehensive fraud detection.

## Dataset

### Original Dataset
- **Source**: [Real or Fake: Fake JobPosting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Content**: Text-based job posting features including job descriptions, requirements, benefits, company profiles, and fraud labels
- **Limitation**: No visual/image features for multimodal analysis

### Enhanced Dataset Features
- **Dataset with Images**: [Multimodal Real / Fake Job Posting Prediction](https://www.kaggle.com/datasets/nithin1729s/multimodal-real-fake-job-posting-prediction)
After running the dataset preparation pipeline, the enhanced dataset includes:
- **Textual Features**: All original text-based attributes
- **Visual Features**: Company profile images scraped from web search results
- **Multimodal Capability**: Combined text and image analysis for improved fraud detection

## Dataset Preparation

Since the original dataset lacked image features essential for multimodal analysis, we developed an automated web scraping system to collect company profile images. This system performs the following operations:

### Image Collection Process

1. **Company Profile Extraction**: Extracts company names and profiles from job postings
2. **Web Search**: Performs automated searches using DuckDuckGo to find company websites
3. **Intelligent Filtering**: Filters out LinkedIn profiles and other irrelevant results
4. **Screenshot Capture**: Takes full-page screenshots of company websites
5. **Organized Storage**: Saves images in labeled folders (0 for legitimate, 1 for fraudulent jobs)

### Key Features of the Image Collection System

- **Parallel Processing**: Uses ThreadPoolExecutor with configurable worker threads (default: 24) for efficient scraping
- **Intelligent Caching**: Implements company-based caching to avoid duplicate downloads for the same company
- **Robust Error Handling**: Handles timeouts, missing elements, and network issues gracefully
- **Anti-Detection Measures**: Uses randomized user agents and headers to avoid blocking
- **Resume Capability**: Supports resuming from specific job IDs in case of interruptions
- **Cookie Management**: Automatically handles cookie consent popups

### Technical Implementation

The image collection system is built using:
- **Selenium WebDriver**: For automated web browsing and screenshot capture
- **ChromeDriver**: Headless Chrome browser for efficient processing
- **Concurrent Futures**: For parallel processing of multiple job postings
- **Threading**: Thread-safe operations with locks for cache and file operations
- **JSON Caching**: Persistent caching system to avoid redundant downloads

### Usage

To run the dataset preparation script:

```bash

pip install -r requirements.txt

cd Dataset\ Preparation 

# Process entire dataset
python duck_duck_go_scrapper_parallelized.py

# Resume from specific job ID
python duck_duck_go_scrapper_parallelized.py 12345
```

### Output Structure

```
images/
├── 0/          # Legitimate job postings
│   ├── 1001.png
│   ├── 1002.png
│   └── ...
└── 1/          # Fraudulent job postings
    ├── 2001.png
    ├── 2002.png
    └── ...
```

To find the image for a particular sample, use the path format: `images/{fraudulent}/{job_id}.png`

## Model Architecture and Training

This section describes the multimodal model design, feature processing, and training pipeline used for fake job posting detection.

### Multimodal Architecture

The system combines **text features** and **image features** into a unified model. The architecture consists of three main components:
<img width="947" height="1678" alt="mermaid-diagram (1)" src="https://github.com/user-attachments/assets/b3395ff7-54fd-4414-b0b7-873120eabc8a" />


#### 1. Text Encoder
- **Input**: Job title, description, requirements, benefits, company profile  
- **Preprocessing**:
  - Lowercasing
  - Stopword removal
  - Tokenization
- **Feature Extraction**:
  - TF-IDF vectorization / pretrained embeddings
- **Output**: Dense feature vector representation of textual data  

#### 2. Image Encoder
- **Input**: Company profile images (website screenshots)  
- **Preprocessing**:
  - Resize to fixed dimensions
  - Normalization
- **Feature Extraction**:
  - Pretrained CNN (e.g., ResNet / EfficientNet)
  - Feature maps flattened into embeddings  
- **Output**: Visual feature vector representation  

#### 3. Fusion Layer
- Concatenates text and image feature vectors  
- Passes combined features through fully connected layers  
- Applies non-linear activation functions (ReLU)  
- Uses dropout for regularization  

#### 4. Classification Head
- Final dense layer with sigmoid / softmax  
- Outputs probability of job being fraudulent  

---

### Training Pipeline

#### Data Splitting
- Train / Validation / Test split  
- Stratified sampling to maintain class balance  

#### Training Configuration
- **Loss Function**: Binary Cross-Entropy  
- **Optimizer**: Adam  
- **Batch Size**: Configurable (e.g., 32 / 64)  
- **Learning Rate**: Tuned using validation set  

#### Training Steps
1. Load textual and image data  
2. Encode text and images separately  
3. Fuse features into a single representation  
4. Train classification model  
5. Validate performance after each epoch  

---

# Results

## Metrics

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.9871 |
| Precision | 0.9045 |
| Recall    | 0.8208 |
| F1 Score  | 0.8606 |
| ROC-AUC   | 0.9758 |


## Classification Report
              precision    recall  f1-score   support

           0       0.99      1.00      0.99      3403
           1       0.90      0.82      0.86       173

    accuracy                           0.99      3576
   macro avg       0.95      0.91      0.93      3576
weighted avg       0.99      0.99      0.99      3576

## Confusion Matrix

|               | Predicted Negative | Predicted Positive |
|---------------|--------------------|--------------------|
| Actual Negative | 3388               | 15                 |
| Actual Positive | 31                 | 142                |

### Summary
- True Negatives (TN): 3388  
- False Positives (FP): 15  
- False Negatives (FN): 31  
- True Positives (TP): 142  
---

### Key Advantages

- Combines **semantic understanding** (text) with **visual credibility cues** (images)  
- Reduces false positives compared to text-only models  
- Scalable to other fraud detection domains  

---

### Limitations

- Image quality depends on scraping success  
- Some companies may not have valid websites  
- Increased computational cost due to multimodal processing  



[*Code can be found here*](https://www.kaggle.com/code/nithin1729s/cv-project-v01) 
