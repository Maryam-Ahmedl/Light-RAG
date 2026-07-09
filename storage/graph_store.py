"""
NetworkX-backed knowledge graph store.

Entities become nodes, relationships become edges. The important detail
copied from LightRAG's design: when the same entity/relationship is seen
again in a later chunk, descriptions are MERGED (concatenated, deduped)
rather than overwritten --- this is what lets the graph accumulate richer
descriptions as more documents are ingested, instead of losing information
to whichever chunk happened to be processed last.
"""

from pathlib import Path

import networkx as nx


class GraphStore:
    def __init__(self, storage_dir: str, name: str = "graph"):
        self.path = Path(storage_dir) / f"{name}.graphml"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.graph = nx.read_graphml(self.path) if self.path.exists() else nx.Graph()

    def save(self):
        nx.write_graphml(self.graph, self.path)

    def upsert_entity(self, name: str, entity_type: str, description: str):
        if name in self.graph.nodes:
            existing_desc = self.graph.nodes[name].get("description", "")
            if description and description not in existing_desc:
                merged = f"{existing_desc} | {description}".strip(" |")
                self.graph.nodes[name]["description"] = merged
        else:
            self.graph.add_node(name, type=entity_type, description=description)

    def upsert_relationship(self, source: str, target: str, description: str):
        # Both endpoints must exist as nodes; create bare-bones ones if the
        # extractor mentioned a relationship without listing an entity.
        for node in (source, target):
            if node not in self.graph.nodes:
                self.graph.add_node(node, type="unknown", description="")

        if self.graph.has_edge(source, target):
            existing_desc = self.graph.edges[source, target].get("description", "")
            if description and description not in existing_desc:
                merged = f"{existing_desc} | {description}".strip(" |")
                self.graph.edges[source, target]["description"] = merged
        else:
            self.graph.add_edge(source, target, description=description)

    def get_entity(self, name: str) -> dict | None:
        if name not in self.graph.nodes:
            return None
        return {"name": name, **self.graph.nodes[name]}

    def get_neighbors(self, name: str, hops: int = 1) -> list[dict]:
        """Return neighbor entities up to N hops away, each with the
        relationship description that connects them back toward `name`."""
        if name not in self.graph.nodes:
            return []

        visited = {name}
        frontier = {name}
        results = []

        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                for neighbor in self.graph.neighbors(node):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
                    edge_desc = self.graph.edges[node, neighbor].get("description", "")
                    results.append(
                        {
                            "name": neighbor,
                            **self.graph.nodes[neighbor],
                            "relationship_to": node,
                            "relationship_description": edge_desc,
                        }
                    )
            frontier = next_frontier

        return results

    def all_entity_names(self) -> list[str]:
        return list(self.graph.nodes)