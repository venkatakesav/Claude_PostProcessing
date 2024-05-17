import transformers
import torch
import re

def process_llama(question_answer_pairs, ocr, output_path):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    with open(output_path, "a") as fp:  # Open file in append mode
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
            \n\n<answer1><Answer></answer1>\n\n
            <answer2>N/A</answer2>\n\n<answer3><Answer></answer3>\n\nEtc.\n\nBegin!"""},
        ]

        prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        answers = re.findall(r'<answer(\d+)>(.*?)</answer\d+>', outputs[0]["generated_text"][len(prompt):])

        print(answers)