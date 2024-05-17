import anthropic
import re

client = anthropic.Anthropic(
    api_key="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",
)

def process_claude(question_answer_pairs, ocr, output_path):
    with open(output_path, "a") as fp:
        message = client.messages.create(
            model="claude-2.1",
            max_tokens=4000,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
                            You will be answering a series of questions based on the OCR text of a document.
                            Here is the OCR:\n\n<ocr>\n{ocr}\n</ocr>\n\n
                            And here are the question-answer pairs in JSON format:\n\n<qa_pairs>\n
                            {question_answer_pairs}\n</qa_pairs>\n\nFor each question, 
                            try to find words and phrases in the OCR text that can be used to answer the question. Minor spelling mistakes in the OCR are okay and
                            should be accounted for. It is critical that you ONLY use information from the OCR text to answer the questions.
                            Give extremely to the point answers, only the key information should be included,
                            NO EXTRA WORDS AT ALL, not even any extra articles, prepositions, and descriptions (other than those mentioned in the circular), ONLY THE ANSWER. 
                            Do not under any circumstances use external information or knowledge to answer.\n\n
                            If a particular question cannot be satisfactorily answered using only the given OCR text, output "N/A" as the answer for that question.\n\n
                            Output your answers inside <answer> tags, numbered to correspond to the question numbers from the JSON. 
                            For example:
                            \n\n<answer1>The purpose of this circular is to ensure the safety of school-going children.</answer1>\n\n
                            <answer2>N/A</answer2>\n\n<answer3>Police verification of drivers and helpers 
                            is necessary to ensure the continued safety of the children.</answer3>\n\nEtc.\n\nBegin!
                            """
                        }
                    ]
                }
            ]
        )
        print(message.content[0].text)
        answers = re.findall(r'<answer(\d+)>(.*?)</answer\d+>', message.content[0].text)
        for num, answer in answers:
            print(f"Answer {num}: {answer}")

        return answers