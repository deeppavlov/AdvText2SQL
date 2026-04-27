import os
from litellm import completion

os.environ["OPENROUTER_API_KEY"] = "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"

response = completion(
    model="openrouter/openai/gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Привет! Напиши короткое приветствие."}
    ]
)

print(response.choices[0].message.content)