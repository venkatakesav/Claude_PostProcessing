import sys
import os
import json

# Add the project root directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.OCR.surya_processor import process_surya

class OCRProcessor:
    def __init__(self, file_directory, ocr_system):
        self.file_directory = file_directory
        self.ocr = None
        if ocr_system == "suryaa":
            self.ocr = self.run_surya_ocr

    def run_surya_ocr(self, image_path):
        return process_surya(image_path)

    def run_tesseract(self, image_path):
        pass

    def run_google_vision_api(self, image_path):
        pass

    def process_directory(self, output_file="output.json"):
        with open(output_file, "a") as fp:  # Open file in append mode
            for file_entry in os.listdir(self.file_directory):
                if file_entry.endswith(".png"):  # Check for PNG files
                    file_path = os.path.join(self.file_directory, file_entry)
                    result = {"file_name": file_entry, "text": self.ocr(file_path)}
                    json.dump(result, fp)  # Write result to file
                    fp.write("\n")  # Add newline separator