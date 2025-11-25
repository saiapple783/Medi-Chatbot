# src/chains/hospital_review_chain.py
import os
from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import RetrievalQA

@lru_cache(maxsize=1)
def get_reviews_vector_chain():
    uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "password")
    
    store = Neo4jVector.from_existing_graph(
        url=uri,
        username=user,
        password=pwd,
        index_name=os.getenv("NEO4J_REVIEWS_INDEX", "reviews"),
        node_label=os.getenv("NEO4J_REVIEWS_NODE", "Review"),
        text_node_property=os.getenv("NEO4J_REVIEWS_TEXT_PROP", "text"),
        embedding_node_property=os.getenv("NEO4J_REVIEWS_EMB_PROP", "embedding"),
    )
    retriever = store.as_retriever(search_type="mmr", k=6)
    llm = ChatOpenAI(model=os.getenv("HOSPITAL_AGENT_MODEL"), temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
