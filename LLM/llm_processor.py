import os
import json

# Add the project root directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.LLM.mistral import process_mistral

class LLMProcessor:
    def __init__(self, claude_annotations, ocr, output_path):
        # Both the claude_annotations, and the ocr are json files
        self.claude_annotations = claude_annotations
        self.ocr = ocr
        self.output_path = output_path

    def run_llama3_8b(self, prompt, json_data):
        # Implementation for Llama3-8B
        pass

    def run_mistral_7b(self):
        process_mistral(self.claude_annotations, self.ocr, self.output_path)
        pass

    def run_claude_2_1(self, prompt, json_data):
        # Implementation for Claude-2.1
        pass

    def run_gpt_3_5(self, prompt, json_data):
        # Implementation for GPT-3.5
        pass

    def process_with_llm(self, prompt, json_data):
        results = {}
        for name, llm_func in self.llm_models.items():
            results[name] = llm_func(prompt, json_data)
        return results
