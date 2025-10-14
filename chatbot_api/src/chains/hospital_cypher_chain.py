import os
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Neo4jVector
from src.langchain_custom.graph_qa.cypher import GraphCypherQAChain

# ---- Env ----
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")  # fix the typo (was NEO4J_Password)

HOSPITAL_QA_MODEL = os.getenv("HOSPITAL_QA_MODEL", "gpt-4o-mini")
HOSPITAL_CYPHER_MODEL = os.getenv("HOSPITAL_CYPHER_MODEL", "gpt-4o-mini")

NEO4J_CYPHER_EXAMPLES_INDEX_NAME = os.getenv("NEO4J_CYPHER_EXAMPLES_INDEX_NAME", "cypher_examples")
NEO4J_CYPHER_EXAMPLES_TEXT_NODE_PROPERTY = os.getenv("NEO4J_CYPHER_EXAMPLES_TEXT_NODE_PROPERTY", "text")
NEO4J_CYPHER_EXAMPLES_NODE_NAME = os.getenv("NEO4J_CYPHER_EXAMPLES_NODE_NAME", "CypherExample")
NEO4J_CYPHER_EXAMPLES_METADATA_NAME = os.getenv("NEO4J_CYPHER_EXAMPLES_METADATA_NAME", "meta")

# ---- Graph ----
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
graph.refresh_schema()

# ---- Embeddings (no proxies kw!) ----
embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))

# ---- Vector index over cypher examples ----
cypher_example_index = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name=NEO4J_CYPHER_EXAMPLES_INDEX_NAME,
    node_label=NEO4J_CYPHER_EXAMPLES_NODE_NAME,  # label, not a property
    text_node_properties=[NEO4J_CYPHER_EXAMPLES_TEXT_NODE_PROPERTY],
    text_node_property=NEO4J_CYPHER_EXAMPLES_TEXT_NODE_PROPERTY,
    embedding_node_property="embedding",
)

cypher_example_retriever = cypher_example_index.as_retriever(search_kwargs={"k": 8})

# ---- Prompts ----
cypher_generation_template = """
Task:
Generate Cypher query for a Neo4j graph database.

Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Schema:
{schema}

Example queries for this schema:
{example_queries}

Warning:
- Never return a review node without explicitly returning all of the properties besides the embedding property
- Make sure to use IS NULL or IS NOT NULL when analyzing missing properties.
- Never include "GROUP BY".
- Alias statements appropriately (e.g., WITH v AS visit, c.billing_amount AS billing_amount)
- Filter denominators to be non-zero before division.

The question is:
{question}
""".strip()

qa_generation_template = """You are an assistant that takes the results from
a Neo4j Cypher query and forms a human-readable response. The provided data is authoritative.
If the provided information is empty ([]), say you don't know the answer. Otherwise answer fully.

Question:
{question}

Query results:
{context}

Helpful Answer:
""".strip()

cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "example_queries", "question"],
    template=cypher_generation_template,
)

qa_generation_prompt = PromptTemplate(
    input_variables=["context", "question"], template=qa_generation_template
)

# ---- Chain ----
hospital_cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOpenAI(model=HOSPITAL_CYPHER_MODEL, temperature=0),
    qa_llm=ChatOpenAI(model=HOSPITAL_QA_MODEL, temperature=0),
    cypher_example_retriever=cypher_example_retriever,
    node_properties_to_exclude=["embedding"],
    graph=graph,
    verbose=True,
    qa_prompt=qa_generation_prompt,
    cypher_prompt=cypher_generation_prompt,
    validate_cypher=True,
    top_k=100,
)
