from OCR.ocr_processor import OCRProcessor

if __name__ == "__main__":
    # The Folder name
    folder = "/data/circulars/DATA/Batches-Split_Images/test_set/from_batch_1_fp"
    processor = OCRProcessor(folder, "suryaa")
    processor.process_directory()