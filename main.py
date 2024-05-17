import json
from OCR.ocr_processor import OCRProcessor
from models.LLM.mistral import process_mistral
from models.LLM.llama3 import process_llama
from models.LLM.claude import process_claude

if __name__ == "__main__":
    # The Folder name
    folder = "/data/circulars/DATA/Batches-Split_Images/test_set/from_batch_1_fp"
    ocr_create_flag = 0
    
    if ocr_create_flag:
        processor = OCRProcessor(folder, "suryaa")
        processor.process_directory()

    with open('qa_dataset/test_batch_1.json', "r") as fp:
        data = json.load(fp)

    with open("output.json", "r") as fp:
        ocr_data = fp.readlines()

    ocr_data_final = dict()
    for entry in ocr_data:
        entry_dict = eval(entry)
        ocr_data_final[entry_dict["file_name"]] = entry_dict["text"]

    for i, entry in enumerate(data[1:]):
        file_name = entry[0]["file_name"]
        print("File Name Is", file_name)
        ocr = ocr_data_final[file_name]
        print("The OCR data is", ocr)

        process_claude(entry[0]["question_answer_pairs"], ocr, "final_output.txt")

        if i == 5:
            break