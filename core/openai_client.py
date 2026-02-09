import os
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def extract_offer(text: str) -> dict:
    prompt = f"""
You are extracting a commercial alcohol offer.
Return JSON ONLY, no explanation.

Fields must match the provided schema.

Text:
{text}
"""

    response = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    return response.choices[0].message.content
