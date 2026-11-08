"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import Entity, ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    graph = nx.DiGraph()
    for result in results:
        for entity in result.entities:
            _add_entity(graph, entity, result.chunk_id)

        for relation in result.relations:
            if not relation.matches_schema():
                continue
            _add_entity(graph, relation.subject, result.chunk_id)
            _add_entity(graph, relation.object, result.chunk_id)
            source, target = relation.subject.node_id, relation.object.node_id
            if not graph.has_edge(source, target):
                graph.add_edge(source, target, type=relation.type, sources=set())
            graph.edges[source, target]["sources"].add(result.chunk_id)

    for _, node in graph.nodes(data=True):
        node["sources"] = sorted(node["sources"])
    for _, _, edge in graph.edges(data=True):
        edge["sources"] = sorted(edge["sources"])
        edge["weight"] = len(edge["sources"])

    _resolve_replacements(graph)
    return graph


def _add_entity(graph: nx.DiGraph, entity: Entity, chunk_id: str) -> None:
    node_id = entity.node_id
    if node_id not in graph:
        graph.add_node(node_id, type=entity.type, name=entity.name, sources=set())
    graph.nodes[node_id]["sources"].add(chunk_id)


def _resolve_replacements(graph: nx.DiGraph) -> None:
    conflicts = 0
    to_remove = []
    for source, target, forward in graph.edges(data=True):
        if source >= target or forward["type"] != "REPLACED_BY":
            continue
        backward = graph.get_edge_data(target, source)
        if backward is None or backward["type"] != "REPLACED_BY":
            continue
        conflicts += 1
        if forward["weight"] <= backward["weight"]:
            to_remove.append((source, target))
        if backward["weight"] <= forward["weight"]:
            to_remove.append((target, source))

    graph.remove_edges_from(to_remove)
    graph.graph["conflicts"] = conflicts
