# app.py

import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from PIL import Image
import io
import logging
import os
import base64
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for all origins (adjust as necessary for security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to the specific domains you want to allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to models
YOLO_MODEL_PATH = "models/malaria_detection/weights/last.pt"
AAM_MODEL_PATH = "aam_model.h5"
DIAGNOSIS_MODEL_PATH = "cnnn_model.h5"

def print_detection_report(results, image_name):
    """Generate a detailed report of the detection results"""
    report = {}
    # Basic Information
    report['analysis_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report['image_analyzed'] = image_name

    # Overall Diagnosis
    report['diagnosis'] = results['diagnosis']
    report['confidence'] = results['confidence']

    # Cell Count Statistics
    total_cells = results['malaria_count'] + results['non_malaria_count']
    infection_rate = (results['malaria_count'] / total_cells) * 100 if total_cells > 0 else 0
    report['cell_statistics'] = {
        'total_cells_detected': total_cells,
        'malaria_infected_cells': results['malaria_count'],
        'non_infected_cells': results['non_malaria_count'],
        'infection_rate': infection_rate
    }

    # Risk Assessment
    if results['diagnosis'] == 'Malaria':
        if infection_rate > 5:
            risk_level = "HIGH RISK"
            recommendation = "Immediate medical attention recommended"
        else:
            risk_level = "MODERATE RISK"
            recommendation = "Further testing recommended"
    else:
        if infection_rate < 0.5:
            risk_level = "LOW RISK"
            recommendation = "Regular monitoring recommended"
        else:
            risk_level = "INCONCLUSIVE"
            recommendation = "Additional testing may be required"
    report['risk_assessment'] = {
        'risk_level': risk_level,
        'recommendation': recommendation,
        'infection_rate': infection_rate
    }

    # Technical Details
    report['technical_details'] = {
        'detection_model': "YOLO v8",
        'classification_model': "AAM",
    }

    return report

def load_models():
    """Load all required models with error handling"""
    models = {}
    try:
        logger.info("Loading YOLO model...")
        models['yolo'] = YOLO(YOLO_MODEL_PATH)
        logger.info("Loading AAM model...")
        models['aam'] = load_model(AAM_MODEL_PATH)
        logger.info(f"AAM model input shape: {models['aam'].input_shape}")
        # If you have a diagnosis model, load it here
        # models['diagnosis'] = load_model(DIAGNOSIS_MODEL_PATH)
        logger.info("All models loaded successfully.")
        return models
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

# Load models at startup
models = load_models()

def load_and_preprocess_image(image_data):
    """Load and preprocess the image from raw bytes"""
    try:
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_np = np.array(image)
        preprocessed_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        return preprocessed_image, image_np
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        raise

def extract_patch(image, box):
    """Extract the patch from the image using the bounding box."""
    try:
        x1, y1, x2, y2, conf, cls = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        patch = image[y1:y2, x1:x2]

        # Resize the patch to 64x64 pixels
        patch_resized = cv2.resize(patch, (64, 64))

        # Normalize the pixel values
        patch_normalized = patch_resized / 255.0

        # Ensure the patch has shape (64, 64, 3)
        logger.info(f"Patch shape: {patch_normalized.shape}")

        return patch_normalized
    except Exception as e:
        logger.error(f"Error extracting patch: {e}")
        return None

def process_image(image_data):
    """Main processing function"""
    try:
        # Load and preprocess image
        preprocessed_image, original_image = load_and_preprocess_image(image_data)

        # Step 1: Cell Detection using YOLO
        logger.info("Detecting cells...")
        results = models['yolo'].predict(source=preprocessed_image, conf=0.5, verbose=False)

        # Extract boxes and patches
        boxes = []
        patches = []
        for result in results:
            for box in result.boxes:
                box_data = box.xyxy[0].cpu().numpy().tolist() + [
                    float(box.conf[0].cpu().numpy()),
                    int(box.cls[0].cpu().numpy())
                ]
                boxes.append(box_data)
                patch = extract_patch(original_image, box_data)
                if patch is not None:
                    patches.append(patch)

        # Step 2: AAM Classification
        if patches:
            patches_array = np.array(patches)  # Should be (num_patches, 64, 64, 3)
            logger.info(f"Patches array shape: {patches_array.shape}")

            # Verify that the patches have the correct shape
            if patches_array.ndim != 4 or patches_array.shape[1:] != (64, 64, 3):
                logger.error(f"Incorrect patches array shape: {patches_array.shape}")
                raise ValueError(f"Incorrect patches array shape: {patches_array.shape}")

            predictions = models['aam'].predict(patches_array)
            patch_predictions = (predictions > 0.5).astype(int).flatten()
            malaria_count = np.sum(patch_predictions)
            non_malaria_count = len(patch_predictions) - malaria_count
        else:
            logger.warning("No patches detected")
            malaria_count = non_malaria_count = 0
            patch_predictions = np.array([])

        # Step 3: Calculate infection rate and determine diagnosis
        total_cells = malaria_count + non_malaria_count
        infection_rate = (malaria_count / total_cells * 100) if total_cells > 0 else 0

        # New logic: Diagnosis based on infection rate
        diagnosis_label = 'Malaria' if infection_rate > 4 else 'Non-Malaria'
        confidence = infection_rate / 100  # Scale infection rate to 0-1 range

        # Step 4: Annotate image
        for box, prediction in zip(boxes, patch_predictions):
            x1, y1, x2, y2, conf, cls = box
            color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
            cv2.rectangle(original_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{'Non-Malaria' if prediction == 0 else 'Malaria'} ({conf:.2f})"
            cv2.putText(original_image, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add overall diagnosis and infection rate
        cv2.putText(original_image, f"Overall: {diagnosis_label} ({infection_rate:.1f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Convert image to base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        logger.info("Image processing complete.")

        results_dict = {
            'diagnosis': diagnosis_label,
            'malaria_count': int(malaria_count),
            'non_malaria_count': int(non_malaria_count),
            'confidence': float(confidence),
            'annotated_image': encoded_image
        }

        # Generate the detection report
        report = print_detection_report(results_dict, "uploaded_image")
        results_dict['report'] = report

        return results_dict

    except Exception as e:
        logger.error(f"Error in process_image: {e}")
        raise

@app.post("/detect-malaria")
async def detect_malaria(file: UploadFile = File(...)):
    """API endpoint to detect malaria from an uploaded image"""
    try:
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Invalid image format")

        image_data = await file.read()
        results = process_image(image_data)
        return JSONResponse(content=results)

    except Exception as e:
        logger.error(f"Failed to process image: {e}")
        raise HTTPException(status_code=500, detail=str(e))
