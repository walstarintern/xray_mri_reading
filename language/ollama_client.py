import ollama

class LocalLLM:
    def __init__(self, model_name="qwen2.5:3b"):
        self.model_name = model_name
        print(f"Connecting to local Ollama model: {self.model_name}...")

    def chat(self, system_prompt, user_question, chat_history=None):
        """
        Sends the prompt and history to the local LLM and returns the response.
        """
        # Start with the system prompt
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add past chat history if it exists (for follow-up questions)
        if chat_history:
            messages.extend(chat_history)
            
        # Add the current user question
        messages.append({"role": "user", "content": user_question})
        
        try:
            # Query the local model (runs completely offline on CPU)
            response = ollama.chat(model=self.model_name, messages=messages)
            return response['message']['content']
        except Exception as e:
            return f"Error connecting to Ollama. Is the Ollama app running? Details: {e}"