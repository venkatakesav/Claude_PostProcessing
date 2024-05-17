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

    with open('filtered_questions.json', "a") as fp:
        for i, entry in enumerate(data[1:]):
            file_name = entry[0]["file_name"]
            print("File Name Is", file_name)
            ocr = ocr_data_final[file_name]
            print("The OCR data is", ocr)

            # filter our the question answer pairs, by checking 
            qa_pairs = entry[0]["question_answer_pairs"]
            qa_final_pairs = []
            q_set = set()
            a_set = set()
            for qa in qa_pairs:
                if qa["question"] in q_set or qa["answer"] in a_set:
                    continue
                else:
                    q_set.add(qa["question"])
                    a_set.add(qa["answer"])
                    qa_final_pairs.append({
                        "question": qa["question"], 
                        "answer": qa["answer"]
                    })

            answers = process_claude(qa_final_pairs, ocr, "final_output.txt")
            for idx, qa in enumerate(qa_final_pairs):
                qa_final_pairs[idx]["answer"] = answers[idx][1]

            json.dump({
                "file_name": file_name,
                "question_answer_pairs": qa_final_pairs
            }, fp)