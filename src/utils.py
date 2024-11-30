import os
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from yfiles_jupyter_graphs import GraphWidget
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)


def _to_df(self: GraphDocument):
    """Convert a list of GraphDocuments to a DataFrame
    GraphDocument is the output from LLMGraphTransformer.convert_to_graph_documents()
    """
    return pd.DataFrame(
        [(rel.source.id, rel.type, rel.target.id) for rel in self.relationships],
        columns=["source", "relationship", "target"],
    )


def _reset(self: Neo4jGraph):
    """Reset the graph by deleting all nodes and refreshing the schema"""
    try:
        self.query("MATCH (n) DETACH DELETE n")
        self.refresh_schema()
        print("Graph reset")
    except Exception as e:
        print(f"Reset failed: {e}")


def _show(self: Neo4jGraph, limit: int = 50):
    """Show the graph in a Jupyter widget using yfiles_jupyter_graphs"""

    cypher = f"MATCH (s)-[r]->(t) RETURN s,r,t LIMIT {limit}"

    session = self._driver.session()
    widget = GraphWidget(graph=session.run(cypher).graph())
    widget.node_label_mapping = "id"
    return widget


def _graph_to_df(self: Neo4jGraph):
    cypher = "MATCH (s)-[r]->(t) RETURN s.id as source, type(r) as relationship, t.id as target"
    return pd.DataFrame(self.query(cypher))


def _get_schemas(self: Neo4jGraph):
    """Get the node and relationship schemas from the graph
    Returns a tuple of two pandas DataFrames:
    - node_schema: list of nodes
    - rel_schema: DataFrame of columns source, type, target
    """
    schema = self.structured_schema
    node_schema = list(schema["node_props"].keys())
    rel_schema = pd.DataFrame([
        {"source": rel["start"], "type": rel["type"], "target": rel["end"]} for rel in schema["relationships"]
    ])
    return node_schema, rel_schema


def _append_graph_documents(
    self: Neo4jGraph, graph_documents: list[GraphDocument], include_source: bool = True, reset: bool = False
):
    if reset:
        self.reset()
    self.add_graph_documents(graph_documents, include_source=include_source, baseEntityLabel=True)


GraphDocument.to_df = _to_df
Neo4jGraph.show = _show
Neo4jGraph.reset = _reset
Neo4jGraph.to_df = _graph_to_df
Neo4jGraph.get_schemas = _get_schemas
Neo4jGraph.append_graph_documents = _append_graph_documents


def convert_to_graph_documents(
    sample_document: list[Document],
    allowed_nodes: list[str],
    allowed_relationships: list[str],
    graph: Neo4jGraph,
    llm: BaseChatModel,
    ignore_tool_usage: bool = False,
):
    """Convert a list of Documents to a list of GraphDocuments
    Filter the nodes and relationships to the allowed ones
    """
    llm_transformer = LLMGraphTransformer(
        llm,
        ignore_tool_usage=ignore_tool_usage,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships,
        node_properties=True,
        relationship_properties=True,
        strict_mode=True,
    )
    graph_documents = llm_transformer.convert_to_graph_documents(sample_document)
    return graph_documents
