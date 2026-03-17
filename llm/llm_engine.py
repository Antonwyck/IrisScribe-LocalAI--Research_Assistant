from llama_cpp import Llama
import os


class LLMEngine:
    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads= 8,#os.cpu_count() or
            n_batch=256,
            verbose=False
        )
#4096
    def generate_stream(self, prompt: str, max_tokens: int = 150):
        stream = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.2,
            repeat_penalty=1.15,
            stop=[
                "</s>",
                "User:",
                "Question:",
                "\n\nUser:",
                "\nWould you like me to",
                "Yes or No?",
                "\nNo.",
                "\nYes."
            ],
            stream=True
        )

        for chunk in stream:
            if "choices" in chunk:
                token = chunk["choices"][0]["text"]
                if token:
                    yield token