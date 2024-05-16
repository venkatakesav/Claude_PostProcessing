import os
from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection import segformer
from surya.model.recognition.model import load_model
from surya.model.recognition.processor import load_processor

IMAGE_DIR = "/data/circulars/DATA/Batches-Split_Images/test_set/from_batch_1_fp"

for file_entry in os.listdir(IMAGE_DIR):
    print("Count is", count)

    IMAGE_PATH = os.path.join(IMAGE_DIR, file_entry)

    image = Image.open(IMAGE_PATH)
    langs = ["en", "hi"]  # Replace with your languages
    det_processor, det_model = segformer.load_processor(), segformer.load_model()
    rec_model, rec_processor = load_model(), load_processor()

    predictions = run_ocr([image], [langs], det_model, det_processor, rec_model, rec_processor)

    # Write the predictions to the file
    fp.write(f"File: {file_entry}\n")
    print(predictions)

    text = ""

    for prediction in predictions[0]:
        for entry in prediction[1]:
            if hasattr(entry, 'text'):
                print("Text is", entry.text)
                text += entry.text + " "

return text