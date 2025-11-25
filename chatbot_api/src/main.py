from fastapi import FastAPI
from src.agents.hospital_rag_agent import hospital_rag_agent_executor
from src.models.hospital_rag_query import HospitalQueryInput, HospitalQueryOutput
from src.utils.async_utils import async_retry

import json

app = FastAPI(
    title="Hospital Chatbot",
    description="Endpoints for a hospital system graph RAG chatbot",
)


@async_retry(max_retries=10, delay=1)
async def invoke_agent_with_retry(query: str):
    """
    Retry the agent if a tool fails to run. This can help when there
    are intermittent connection issues to external APIs.
    """
    return await hospital_rag_agent_executor.ainvoke({"input": query})


@app.get("/")
async def get_status():
    return {"status": "running"}


@app.post("/hospital-rag-agent", response_model=HospitalQueryOutput)
async def ask_hospital_agent(query: HospitalQueryInput) -> HospitalQueryOutput:
    """
    Call the hospital RAG agent and normalize the response into a
    readable, well-structured format.

    - If the agent's output is a JSON array of hospitals, pretty-print it.
    - Otherwise, return the raw text output.
    """
    agent_result = await invoke_agent_with_retry(query.text)

    # LangChain AgentExecutor with return_intermediate_steps=True
    # typically returns: {"input": ..., "output": ..., "intermediate_steps": [...]}
    raw_output = agent_result.get("output", "")
    intermediate_steps = [
        str(s) for s in agent_result.get("intermediate_steps", [])
    ]

    pretty_output = raw_output

    # Try to parse as JSON array of hospital objects:
    # [
    #   {"name": "...", "address": "...", "phone": "...", "note": "..."},
    #   ...
    # ]
    try:
        data = json.loads(raw_output)
        if isinstance(data, list) and all(isinstance(h, dict) for h in data):
            lines = []

            if data:
                lines.append("Here are the hospitals I found:\n")
            else:
                lines.append("I couldn't find any hospitals for that query.")

            for idx, h in enumerate(data, start=1):
                name = h.get("name", "Unknown hospital")
                address = h.get("address")
                phone = h.get("phone")
                note = h.get("note")

                lines.append(f"{idx}. {name}")
                if address:
                    lines.append(f"   {address}")
                if phone:
                    lines.append(f"   Phone: {phone}")
                if note:
                    lines.append(f"   {note}")
                lines.append("")  # blank line between entries

            pretty_output = "\n".join(lines).strip()
    except (json.JSONDecodeError, TypeError):
        # Not JSON; leave pretty_output as the original raw_output
        pass

    return HospitalQueryOutput(
        input=query.text,
        output=pretty_output,
        intermediate_steps=intermediate_steps,
    )
