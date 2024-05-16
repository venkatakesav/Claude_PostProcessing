import os
from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection import segformer
from surya.model.recognition.model import load_model
from surya.model.recognition.processor import load_processor

def process_surya(image_path): 
    image = Image.open(image_path)
    langs = ["en", "hi"]  # Replace with your languages
    det_processor, det_model = segformer.load_processor(), segformer.load_model()
    rec_model, rec_processor = load_model(), load_processor()

    predictions = run_ocr([image], [langs], det_model, det_processor, rec_model, rec_processor)

    text = ""

    for prediction in predictions[0]:
        for entry in prediction[1]:
            if hasattr(entry, 'text'):
                print("Text is", entry.text)
                text += entry.text + " "

    return text