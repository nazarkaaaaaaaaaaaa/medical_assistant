import os
import json
import re  # <-- Добавлен импорт для очистки текста
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# --- Инициализация и Настройки ---
load_dotenv()

app = FastAPI()
# Убедитесь, что папка static существует
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set. Please set it in .env file.")

groq_client = Groq(api_key=groq_api_key)

# Актуальная модель Groq
GROQ_MODEL = "qwen/qwen3-32b"


# --- Pydantic модель для входных данных ---
class DiagnosisRequest(BaseModel):
    symptoms: str
    age: int
    gender: str
    history: str = "No significant medical history provided."


# --- Функции очистки и агентов ---

def clean_agent_output(text: str) -> str:
    """
    Удаляет внутренние размышления (<think>...</think>) и английский/лишний текст,
    который LLM генерирует перед началом ответа на русском.
    """
    # 1. Удаление блока <think> (если есть)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # 2. Поиск начала первого значимого русского раздела/заголовка
    # Ищем: Диагноз, Лечение, Положительные аспекты или любую заглавную русскую букву.
    first_section = re.search(r'([А-ЯЁ]|\n#[ \wА-ЯЁа-яё]+)', text)

    if first_section:
        # Обрезаем текст до найденного начала
        return text[first_section.start():].strip()

    # Если ничего не найдено, возвращаем текст как есть
    return text.strip()


def run_agent(system_prompt: str, user_prompt: str) -> str:
    """Общая функция для выполнения запроса к Groq API."""
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=GROQ_MODEL,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        error_msg = f"Error calling Groq API: {e}"
        print(error_msg)
        return f"ERROR: {error_msg}"


def agent_diagnostician(data: DiagnosisRequest) -> str:
    """Агент 1: Ставит диагноз."""
    system_prompt = (
        "You are a Senior Medical Diagnostician. Your task is to analyze patient symptoms "
        "and provide the MOST PROBABLE differential diagnosis and its justification in Russian. "
        "Your ENTIRE response MUST be ONLY the final medical report in Russian, strictly avoiding "
        "any internal thoughts, pre-amble, or English text. "
        "Structure the response with headings and lists using Markdown (e.g., # Заголовок, **Жирный текст**, * Список)."
    )
    user_prompt = (
        f"Patient Data:\n"
        f"- Symptoms: {data.symptoms}\n"
        f"- Age: {data.age}\n"
        f"- Gender: {data.gender}\n"
        f"- Medical History: {data.history}\n\n"
        f"Provide the diagnosis in Russian now."
    )
    return run_agent(system_prompt, user_prompt)


def agent_therapist(diagnosis_report: str) -> str:
    """Агент 2: Составляет лечение на основе диагноза."""
    system_prompt = (
        "You are a skilled Treatment Planner. Your goal is to formulate a safe and comprehensive "
        "treatment plan based ONLY on the provided diagnosis. The plan must include three sections: "
        "'# Медикаментозное лечение', '# Стиль жизни и профилактика', and '# План наблюдения'. "
        "Your ENTIRE response MUST be ONLY the final treatment plan in Russian. "
        "DO NOT include any thoughts or English text. Use Markdown for structure (e.g., # Заголовок, * Список)."
    )
    user_prompt = f"Diagnosis to be treated (in Russian):\n---\n{diagnosis_report}\n---\nCreate the treatment plan in Russian now."
    return run_agent(system_prompt, user_prompt)


def agent_monitor(diagnosis_report: str, treatment_plan: str) -> str:
    """Агент 3: Следит/проверяет лечение."""
    system_prompt = (
        "You are a Treatment Monitor and Safety Checker. Your goal is to review the proposed treatment plan "
        "for safety, potential risks, and general suitability against the initial diagnosis. "
        "Your review must be structured with '# Положительные аспекты', '# Предложения по улучшению', and '# Заключение'. "
        "Your ENTIRE response MUST be ONLY the final review in Russian. "
        "DO NOT include any thoughts or English text. Use Markdown for structure."
    )
    user_prompt = (
        f"Diagnosis (Russian):\n---\n{diagnosis_report}\n---\n"
        f"Proposed Treatment Plan (Russian):\n---\n{treatment_plan}\n---\n"
        f"Review the plan and provide feedback in Russian."
    )
    return run_agent(system_prompt, user_prompt)


# --- Главный API маршрут ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/diagnose_and_treat")
async def diagnose_and_treat(request_data: DiagnosisRequest):
    try:
        # 1. Запуск Агента Диагноста
        diagnosis_raw = agent_diagnostician(request_data)
        diagnosis_report = clean_agent_output(diagnosis_raw)  # <-- Очистка

        if "ERROR:" in diagnosis_report:
            raise Exception(diagnosis_report.replace("ERROR: ", ""))

        # 2. Запуск Агента Терапевта (на основе диагноза)
        treatment_raw = agent_therapist(diagnosis_report)
        treatment_plan = clean_agent_output(treatment_raw)  # <-- Очистка

        if "ERROR:" in treatment_plan:
            raise Exception(treatment_plan.replace("ERROR: ", ""))

        # 3. Запуск Агента Монитора (на основе диагноза и лечения)
        monitoring_raw = agent_monitor(diagnosis_report, treatment_plan)
        monitoring_feedback = clean_agent_output(monitoring_raw)  # <-- Очистка

        return {
            "success": True,
            "diagnosis": diagnosis_report,
            "treatment": treatment_plan,
            "monitoring_feedback": monitoring_feedback,
        }

    except Exception as e:
        error_message = f"An orchestrated error occurred: {str(e)}"
        print(error_message)
        return {"success": False, "error": error_message}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)