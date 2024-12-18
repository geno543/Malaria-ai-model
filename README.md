# Malaria Cell Detection API

A FastAPI-based application that uses deep learning models (YOLO and AAM) to detect and classify malaria-infected cells in microscopic images.

## Features

- Cell detection using YOLOv8
- Malaria infection classification using AAM (Attention Augmented Model)
- Real-time image processing and analysis
- Detailed detection reports with risk assessment
- REST API endpoints for easy integration

## Requirements

- Python 3.8+
- FastAPI
- OpenCV
- TensorFlow
- PyTorch
- Ultralytics YOLO
- Pillow

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd malaria-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the required model files:
- Place YOLO model in `models/malaria_detection/weights/last.pt`
- Place AAM model in `aam_model.h5`
- Place CNN model in `cnnn_model.h5`

## Usage

1. Start the FastAPI server:
```bash
uvicorn malaria_3:app --reload
```

2. Send POST requests to `/detect-malaria` endpoint with an image file.

## API Endpoints

### POST /detect-malaria
- Accepts image files (JPEG, PNG)
- Returns detection results including:
  - Diagnosis (Malaria/Non-Malaria)
  - Cell counts
  - Confidence score
  - Annotated image
  - Detailed report with risk assessment

## Response Format

```json
{
    "diagnosis": "Malaria/Non-Malaria",
    "malaria_count": int,
    "non_malaria_count": int,
    "confidence": float,
    "annotated_image": "base64_encoded_string",
    "report": {
        "analysis_date": "timestamp",
        "risk_assessment": {
            "risk_level": "string",
            "recommendation": "string"
        },
        ...
    }
}
```

## License

[Your chosen license]
