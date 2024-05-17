import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda:0"  # the device to load the model onto

def process_mistral(question_answer_pairs, ocr, output_path):
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    
    model.to(device)  # Load model to the device

    with open(output_path, "a") as fp:  # Open file in append mode
        # Find the corresponding entry in the claude_annotations
        messages = [
            {"role": "user", 
            "content": f"""You will be answering a series of questions based on the OCR text of a document.
            Here is the OCR:\n\n<ocr>\n{ocr}\n</ocr>\n\n
            And here are the question-answer pairs in JSON format:\n\n<qa_pairs>\n
            {question_answer_pairs}\n</qa_pairs>\n\nFor each question, 
            try to find words and phrases in the OCR text that can be used to answer the question. Minor spelling mistakes in the OCR are okay and
            should be accounted for. It is critical that you ONLY use information from the OCR text to answer the questions.
            Give extremely to the point answers, only the key information should be included,
            NO EXTRA WORDS AT ALL, not even any extra articles, prepositions, and descriptions, ONLY THE ANSWER. 
            Do not under any circumstances use external information or knowledge to answer.\n\n
            If a particular question cannot be satisfactorily answered using only the given OCR text, output "N/A" as the answer for that question.\n\n
            Output your answers inside <answer> tags, numbered to correspond to the question numbers from the JSON. 
            For example:
            \n\n<answer1>The purpose of this circular is to ensure the safety of school-going children.</answer1>\n\n
            <answer2>N/A</answer2>\n\n<answer3>Police verification of drivers and helpers 
            is necessary to ensure the continued safety of the children.</answer3>\n\nEtc.\n\nBegin!"""},
        ]

        print("Message being sent is", messages)

        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        inputs = tokenizer(str(messages), return_tensors="pt", truncation=True, padding=True)

        inputs = {key: val.to(device) for key, val in inputs.items()}

        generated_ids = model.generate(**inputs, max_new_tokens=1000, do_sample=True)
        decoded_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Extract answers using regex
        answers = re.findall(r'<answer(\d+)>(.*?)</answer\d+>', decoded_output)

        print("Answers", answers)

        modified_qa_pairs = []
        for num, answer in answers:
            question = next(q["question"] for q in question_answer_pairs if q["id"] == int(num))
            modified_qa_pairs.append({"id": int(num), "question": question, "answer": answer.strip()})

        result = {"file_name": file_name, "modified_question_answer_pairs": modified_qa_pairs}
        json.dump(result, fp)
        fp.write("\n")  # Add newline separator