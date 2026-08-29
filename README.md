# Image Quality Assessment System

## Overview

This project is an AI-based Image Quality Assessment system that checks the visual quality of an image and identifies common quality problems.

The system combines traditional computer-vision features with a Convolutional Autoencoder and a Random Forest classifier. It classifies an image into one of three categories:

- ACCEPTABLE
- DEGRADED
- DEFECTIVE

The application has a React frontend, FastAPI backend, SQLite database, trained ML models, and Docker support.

---

## Features

- Upload and preview an image
- Analyze image quality
- Generate a quality score
- Classify images as ACCEPTABLE, DEGRADED, or DEFECTIVE
- Detect blur, underexposure, overexposure, noise, and visual defects
- Show issue severity and confidence
- Store previous analyses
- View analysis history
- Health check endpoint
- Docker/Docker Compose deployment

---

## System Workflow

```text
Image Upload
     |
     v
Image Preprocessing
     |
     +------------------------+
     |                        |
     v                        v
Computer Vision Features   Autoencoder
     |                        |
     |                  Reconstruction Error
     |                        |
     +-----------+------------+
                 |
                 v
          12 Feature Values
                 |
                 v
        Random Forest Classifier
                 |
       +---------+---------+
       |         |         |
       v         v         v
  ACCEPTABLE  DEGRADED  DEFECTIVE
                 |
                 v
        Quality Score + Issues
                 |
                 v
           SQLite History