import os
import json
import base64
import pandas as pd
from typing import Dict, Any
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def extract_offer(text: str) -> dict:
    prompt = f"""
    You are extracting a commercial alcohol offer.
    Return JSON ONLY, no explanation.

    Extract all possible fields from the text. If a field is not found, return null.

    Fields to extract:
    - product_name
    - product_key
    - brand
    - category
    - sub_category
    - packaging
    - packaging_raw
    - bottle_or_can_type
    - unit_volume_ml
    - units_per_case
    - cases_per_pallet
    - quantity_case
    - gift_box
    - refillable_status
    - currency
    - price_per_unit
    - price_per_unit_eur
    - price_per_case
    - price_per_case_eur
    - fx_rate
    - fx_date
    - alcohol_percent
    - origin_country
    - supplier_country
    - incoterm
    - location
    - lead_time
    - moq_cases
    - valid_until
    - best_before_date
    - vintage
    - supplier_reference
    - ean_code
    - label_language
    - product_reference

    Text:
    {text}
    """

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        content = response.choices[0].message.content
        return json.loads(content)

    except Exception as e:
        print(f"Error extracting from text: {e}")
        return {}


async def extract_from_file(file_path: str, content_type: str) -> Dict[str, Any]:
    """Extract data from uploaded files using OpenAI"""

    try:
        text_content = ""

        if content_type in [
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        ]:
            try:
                df = pd.read_excel(file_path)
                text_content = df.to_string()

                if not text_content.strip():
                    text_content = "\n".join([
                        f"Columns: {', '.join(df.columns.tolist())}",
                        f"First few rows:\n{df.head().to_string()}"
                    ])
            except Exception as e:
                text_content = f"Excel file - error reading: {str(e)}"

        elif content_type == 'application/pdf':
            try:
                import PyPDF2
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_content = ""
                    for page in pdf_reader.pages:
                        text_content += page.extract_text()
            except Exception as e:
                text_content = f"PDF file - error reading: {str(e)}"

        elif 'image' in content_type:
            # For images
            try:
                with open(file_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": """Extract commercial alcohol product information from this image. 
                                Look for: product name, brand, packaging type, volume, price, currency, alcohol percentage, 
                                supplier information, quantities, and any other relevant commercial details."""},
                                {
                                    "type": "image_url",
                                    "image_url": f"data:{content_type};base64,{base64_image}",
                                },
                            ],
                        }
                    ],
                    max_tokens=1000,
                )
                return await extract_offer(response.choices[0].message.content)

            except Exception as e:
                print(f"Error extracting from image: {e}")
                return {}

        else:
            # For text files
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text_content = file.read()
            except:
                with open(file_path, 'r', encoding='latin-1') as file:
                    text_content = file.read()

        if text_content:
            return await extract_offer(text_content)
        else:
            return {}

    except Exception as e:
        print(f"Error extracting from file: {e}")
        return {}


def parse_buffer_data(buffer_data: dict) -> bytes:
    """Convert make.com buffer data to bytes"""
    if isinstance(buffer_data, dict) and buffer_data.get('type') == 'Buffer':
        return bytes(buffer_data['data'])
    elif isinstance(buffer_data, dict) and 'data' in buffer_data:
        try:
            if isinstance(buffer_data['data'], str):
                return base64.b64decode(buffer_data['data'])
            elif isinstance(buffer_data['data'], list):
                return bytes(buffer_data['data'])
        except:
            pass
    elif isinstance(buffer_data, str):
        try:
            return base64.b64decode(buffer_data)
        except:
            pass

    return b''