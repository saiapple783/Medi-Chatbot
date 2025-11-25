import os
import json
from typing import Any

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, tool, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.tools.wait_times import (
    get_current_wait_times,
    get_most_available_hospital,
)

from openai import OpenAI


client = OpenAI() 

HOSPITAL_AGENT_MODEL = os.getenv("HOSPITAL_AGENT_MODEL", "gpt-4o-mini")

agent_chat_model = ChatOpenAI(
    model=HOSPITAL_AGENT_MODEL,
    temperature=0,
)



@tool
def get_hospital_wait_time(hospital: str) -> str:
    """
    Use when asked about the current wait time at a specific hospital
    in this demo. Input is ONLY the hospital name (no extra words).
    """
    try:
        return get_current_wait_times(hospital)
    except Exception as e:
        return f"wait_time_unavailable: {type(e).__name__}: {e}"


@tool
def find_most_available_hospital() -> dict[str, float]:
    """
    Use when asked which hospital currently has the shortest wait time
    in this demo. Takes no arguments.
    """
    try:
        return get_most_available_hospital(None)
    except Exception as e:
        return {"error": f"most_available_unavailable: {type(e).__name__}: {e}"}


@tool
def live_hospital_search(query: str) -> str:
    """
    Use this to get up-to-date lists of hospitals for a given location, ZIP,
    or radius using OpenAI's web search.
    The query string should include all relevant details
    (e.g. "hospitals in Irving, TX within 25 miles").

    Returns:
        A JSON array (as string) of hospital objects:
        [
          {
            "name": "...",
            "address": "...",
            "phone": "...",
            "note": "..."
          },
          ...
        ]
    """
    
    response = client.responses.create(
        model=HOSPITAL_AGENT_MODEL,
        input=(
            "Use web search to find hospitals matching this request:\n"
            f"{query}\n\n"
            "Return ONLY a valid JSON array of hospital objects with keys: "
            "name, address, phone, note (omit unknown fields). "
            "No markdown, no extra commentary."
        ),
        tools=[{"type": "web_search"}],
    )

    
    text = ""
    if hasattr(response, "output_text") and response.output_text:
        text = response.output_text
    else:
        chunks = []
        for item in getattr(response, "output", []):
            for c in getattr(item, "content", []):
                # Newer SDK: c.type == "output_text"
                if getattr(c, "type", None) == "output_text":
                    chunks.append(c.text)
                # Older style: just text field
                elif hasattr(c, "text"):
                    chunks.append(c.text)
        text = "".join(chunks).strip()

    if not text:
        return "[]"

    
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return json.dumps(data)
    except Exception:
        pass

    return "[]"


agent_tools = [
    get_hospital_wait_time,
    find_most_available_hospital,
    live_hospital_search,
]



agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a strict Hospital & Healthcare Assistant for a demo application.

HARD RULES:

1. SCOPE:
   You ONLY answer questions related to:
   - Hospitals, clinics, urgent care, emergency rooms
   - Doctors/physicians and healthcare providers
   - Insurance / networks as they relate to hospitals
   - Hospital quality, wait times, patient experience, services
   - Finding hospitals by city, state, ZIP code, or approximate radius

2. OUT-OF-SCOPE:
   If the user asks about anything outside this scope
   (programming, generic coding help, travel, sports, math puzzles, small-talk),
   respond with EXACTLY:
   "I can only help with hospital and healthcare information."

3. FINDING / LISTING HOSPITALS (IMPORTANT):
   When the user asks you to FIND or LIST hospitals, such as:
     - "Hospitals in {{city}}, {{state}}"
     - "Hospitals near {{ZIP}}"
     - "Hospitals within {{N}} miles of {{location}}"
     - "List hospitals around {{location}}"
   you MUST:
     a) Call the `live_hospital_search` tool with the full user query.
     b) Use ONLY its JSON output for your final answer.
   The final answer for these queries MUST be ONLY a valid JSON array
   of hospital objects:

        [
          {{
            "name": "Hospital Name",
            "address": "Street, City, State, ZIP",
            "phone": "+1-000-000-0000",
            "note": "short description or type if known"
          }}
        ]

   - You may include multiple such objects.
   - If you do not know an exact field for a hospital, omit that field.
   - If no hospitals can be found, return [].
   - Do NOT include markdown or extra prose with that JSON.

4. TOOL USAGE:
   - `live_hospital_search`:
       * Required for any hospital-finding / listing queries.
   - `get_hospital_wait_time`:
       * Only for current wait time at a specific named hospital in this demo.
   - `find_most_available_hospital`:
       * Only for "which hospital has the shortest wait time" in this demo.
   - Do NOT use wait-time tools to search for hospitals by geography.

5. STYLE:
   - For hospital-list queries: JSON only (as above).
   - For other valid in-scope answers: concise and clear.
            """,
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


hospital_rag_agent = create_openai_tools_agent(
    llm=agent_chat_model,
    tools=agent_tools,
    prompt=agent_prompt,
)

hospital_rag_agent_executor = AgentExecutor(
    agent=hospital_rag_agent,
    tools=agent_tools,
    verbose=True,
    return_intermediate_steps=True,
)
