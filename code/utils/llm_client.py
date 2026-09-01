import time
import tiktoken
from openai import OpenAI
from config.settings import (
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL,
    OPENAI_API_KEY, OPENAI_BASE_URL,
    DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, DEFAULT_TIMEOUT,
    MODELS
)


class LLMClient:
    def __init__(self, model=DEFAULT_MODEL):
        self.model = model
        provider = MODELS.get(model, {}).get("provider", "dashscope")
        if provider == "dashscope":
            self.client = OpenAI(
                api_key=DASHSCOPE_API_KEY,
                base_url=DASHSCOPE_BASE_URL
            )
        else:
            self.client = OpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL
            )

    def chat(self, messages, temperature=DEFAULT_TEMPERATURE,
             max_tokens=DEFAULT_MAX_TOKENS, timeout=DEFAULT_TIMEOUT):
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
        latency = time.time() - start_time
        content = response.choices[0].message.content
        usage = response.usage
        return {
            "content": content,
            "model": response.model,
            "latency": latency,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        }

    def simple_chat(self, question, system_prompt=None, temperature=DEFAULT_TEMPERATURE):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        return self.chat(messages, temperature=temperature)

    def chat_stream(self, messages, temperature=DEFAULT_TEMPERATURE,
                    max_tokens=DEFAULT_MAX_TOKENS):
        ttft = None
        start_time = time.time()
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        chunks = []
        token_times = []
        prev_time = None
        for chunk in stream:
            if chunk.choices[0].delta.content:
                current_time = time.time()
                if ttft is None:
                    ttft = current_time - start_time
                if prev_time:
                    token_times.append(current_time - prev_time)
                prev_time = current_time
                chunks.append(chunk.choices[0].delta.content)

        full_content = "".join(chunks)
        avg_tpot = (sum(token_times) / len(token_times) * 1000) if token_times else 0
        return {
            "content": full_content,
            "ttft": ttft,
            "avg_tpot_ms": avg_tpot,
            "token_count": len(token_times) + 1,
            "total_latency": time.time() - start_time,
        }

    @staticmethod
    def count_tokens(text, model="cl100k_base"):
        try:
            encoding = tiktoken.get_encoding(model)
            return len(encoding.encode(text))
        except Exception:
            return len(text) // 4


def call_llm(question, model=DEFAULT_MODEL, system_prompt=None, temperature=DEFAULT_TEMPERATURE):
    client = LLMClient(model=model)
    result = client.simple_chat(question, system_prompt=system_prompt, temperature=temperature)
    return result["content"]


def call_llm_with_messages(messages, model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE):
    client = LLMClient(model=model)
    result = client.chat(messages, temperature=temperature)
    return result["content"]
