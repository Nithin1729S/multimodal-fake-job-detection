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