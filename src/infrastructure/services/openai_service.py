import json
from typing import AsyncGenerator, Optional

from openai import AsyncOpenAI

from src.domain.entities.doctors import DoctorWithDetailsEntity


class OpenAIService:
    def __init__(self, api_key: str):
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = "gpt-4o-mini"

    def _format_doctors_for_prompt(self, doctors: list[DoctorWithDetailsEntity]) -> str:
        if not doctors:
            return "\n\nNO DOCTORS CURRENTLY AVAILABLE ON OUR PLATFORM."

        doctors_by_spec: dict[str, list[DoctorWithDetailsEntity]] = {}
        for doctor in doctors:
            spec = doctor.specialization_name
            if spec not in doctors_by_spec:
                doctors_by_spec[spec] = []
            doctors_by_spec[spec].append(doctor)

        lines = ["\n\nAVAILABLE DOCTORS IN OUR PLATFORM:"]
        lines.append(f"(Specializations available: {', '.join(doctors_by_spec.keys())})")
        for spec, spec_doctors in doctors_by_spec.items():
            lines.append(f"\n{spec}:")
            for doc in spec_doctors:
                bio_preview = doc.bio[:100] if doc.bio else "No bio available"
                lines.append(
                    f"  - Dr. {doc.full_name} (ID: {doc.id}) | "
                    f"Rating: {doc.rating}/5 | "
                    f"Experience: {doc.experience_years} years | "
                    f"Bio: {bio_preview}..."
                )
        return "\n".join(lines)

    def _get_system_prompt(
        self, doctors: Optional[list[DoctorWithDetailsEntity]] = None
    ) -> str:
        doctors_section = self._format_doctors_for_prompt(doctors) if doctors else ""

        return f"""You are an AI medical assistant for a healthcare platform called MedCare. Your role is to:

1. Listen to patient symptoms and concerns with empathy
2. Ask 1-2 brief clarifying questions to understand their condition better
3. Provide general health information (NOT diagnoses)
4. Recommend which type of medical specialist they should consult
5. Assess urgency level (low, medium, high, emergency)
6. Recommend specific doctors from our platform OR provide external resources if no matching doctors available

IMPORTANT GUIDELINES:
- Never provide definitive diagnoses
- Always recommend consulting a real doctor
- For emergency symptoms (chest pain, difficulty breathing, severe bleeding, etc.), immediately advise seeking emergency care
- Be compassionate and professional
- ASK MAXIMUM 2-3 QUESTIONS before making a recommendation. After 2 exchanges, you MUST provide doctor recommendations
- When recommending doctors, prefer those with higher ratings and more experience
- Recommend up to 3 doctors that best match the patient's needs
- ALWAYS include a JSON recommendation block after your conversational response once you have basic symptom information
{doctors_section}

RESPONSE FORMAT:
After gathering basic information (usually after 1-2 questions), include both a conversational response AND a JSON block.

CASE 1 - If matching doctors ARE available on our platform:
```json
{{
    "recommendation": true,
    "specialization": "Dentistry",
    "confidence": 0.85,
    "urgency": "medium",
    "reasoning": "Based on the described symptoms...",
    "has_platform_doctors": true,
    "recommended_doctor_ids": [1, 2, 3],
    "recommended_doctors": [
        {{"id": 1, "name": "Dr. John Smith", "specialization": "Dentistry", "rating": 4.8, "experience_years": 10}}
    ]
}}
```

CASE 2 - If NO matching doctors available on our platform (or platform has no doctors):
Provide helpful external resources. Include search links and general guidance.
```json
{{
    "recommendation": true,
    "specialization": "Cardiology",
    "confidence": 0.80,
    "urgency": "high",
    "reasoning": "Based on your symptoms, you should see a cardiologist...",
    "has_platform_doctors": false,
    "external_resources": [
        {{
            "name": "Find Cardiologists Near You",
            "type": "search",
            "url": "https://www.google.com/search?q=cardiologist+near+me",
            "description": "Search for cardiologists in your area"
        }},
        {{
            "name": "Zocdoc - Book Cardiologist",
            "type": "booking",
            "url": "https://www.zocdoc.com/search?dr_specialty=cardiologist",
            "description": "Find and book appointments with cardiologists"
        }},
        {{
            "name": "Healthgrades",
            "type": "directory",
            "url": "https://www.healthgrades.com/cardiology-directory",
            "description": "Doctor reviews and ratings"
        }}
    ],
    "emergency_contacts": {{
        "emergency": "911",
        "poison_control": "1-800-222-1222",
        "mental_health": "988"
    }},
    "general_advice": "While we don't have cardiologists on our platform yet, I recommend using the links above to find a specialist near you. If symptoms worsen, please seek emergency care immediately."
}}
```

For the FIRST message from a patient, ask 1-2 clarifying questions without the JSON block.
For the SECOND or THIRD message, you MUST include the JSON block with recommendations (either platform doctors OR external resources)."""

    async def chat_stream(
            self,
            messages: list[dict],
            doctors: Optional[list[DoctorWithDetailsEntity]] = None,
            temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        system_message = {"role": "system", "content": self._get_system_prompt(doctors)}
        all_messages = [system_message] + messages

        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=all_messages,
            temperature=temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def chat(
            self,
            messages: list[dict],
            doctors: Optional[list[DoctorWithDetailsEntity]] = None,
            temperature: float = 0.7,
    ) -> str:
        system_message = {"role": "system", "content": self._get_system_prompt(doctors)}
        all_messages = [system_message] + messages

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=all_messages,
            temperature=temperature,
        )

        return response.choices[0].message.content

    async def analyze_symptoms(
            self,
            symptoms: str,
            conversation_history: list[dict],
    ) -> dict:
        analysis_prompt = f"""Based on the following conversation about symptoms, provide a final analysis.

Symptoms initially described: {symptoms}

Provide your response in this exact JSON format:
{{
    "recommended_specialization": "string - medical specialty name",
    "confidence": 0.0-1.0,
    "urgency": "low|medium|high|emergency",
    "summary": "brief summary of the consultation",
    "key_symptoms": ["list", "of", "symptoms"],
    "suggested_questions_for_doctor": ["questions the patient should ask"]
}}"""

        messages = conversation_history + [{"role": "user", "content": analysis_prompt}]

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        return json.loads(response.choices[0].message.content)
