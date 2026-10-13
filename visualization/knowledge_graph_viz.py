"""Render TrippinnI's serialized healthcare knowledge graph with Graphviz.

Input may be a bare HealthcareKnowledgeGraph.to_dict() payload or the
healthcare_context.json report that nests it under "knowledge_graph".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from graphviz import Digraph


def load_graph(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    graph = payload.get("knowledge_graph", payload)
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("edges"), list):
        raise ValueError(
            "Expected 'nodes' and 'edges' either at the JSON root or under "
            "'knowledge_graph'."
        )
    return graph


def build_graph(payload: dict[str, Any], output_format: str = "svg") -> Digraph:
    """Build a Graphviz diagram from TrippinnI's actual graph schema."""
    dot = Digraph(
        name="TrippinnI_Healthcare_Knowledge_Graph",
        format=output_format,
        graph_attr={
            "rankdir": "LR",
            "label": "TrippinnI Healthcare Knowledge Graph",
            "labelloc": "t",
            "fontsize": "18",
            "fontname": "Arial",
            "nodesep": "0.25",
            "ranksep": "0.65",
            "splines": "true",
            "overlap": "false",
        },
        node_attr={"fontname": "Arial", "fontsize": "9", "margin": "0.08,0.04"},
        edge_attr={"fontname": "Arial", "fontsize": "7", "arrowsize": "0.65"},
    )

    nodes = {node["id"]: node for node in payload["nodes"] if "id" in node}

    # Some current graph edges point at concept IDs that the builder has not
    # materialized as nodes. Add display-only placeholders to preserve edges.
    for edge in payload["edges"]:
        for endpoint in (edge.get("source"), edge.get("target")):
            if endpoint and endpoint not in nodes:
                nodes[endpoint] = {
                    "id": endpoint,
                    "type": "concept",
                    "properties": {"display_only_placeholder": True},
                }

    for node_id, node in nodes.items():
        node_type = str(node.get("type", "unknown"))
        props = node.get("properties") or {}

        if node_type == "table":
            label = f"TABLE\\n{props.get('table', node_id)}"
            fill, shape = "lightblue", "box"
        elif node_type == "field":
            table = str(props.get("table", ""))
            prefix = f"column:{table}."
            column = node_id[len(prefix):] if node_id.startswith(prefix) else node_id
            roles = props.get("roles") or []
            label = f"{table}.{column}"
            if roles:
                label += "\\nroles: " + ", ".join(map(str, roles))
            fill, shape = "lightgoldenrod1", "box"
        else:
            label = node_id.replace("concept:", "CONCEPT\\n")
            fill, shape = "palegreen", "ellipse"

        dot.node(
            node_id,
            label=label,
            fillcolor=fill,
            shape=shape,
            style="rounded,filled",
            tooltip=json.dumps(props, ensure_ascii=False, default=str),
        )

    for edge in payload["edges"]:
        source, target = edge.get("source"), edge.get("target")
        if not source or not target:
            continue
        label = str(edge.get("relation", "RELATED_TO"))
        confidence = edge.get("confidence")
        if confidence is not None:
            try:
                label += f" ({float(confidence):.2f})"
            except (TypeError, ValueError):
                pass
        dot.edge(source, target, label=label)

    return dot


def render_graph(
    input_path: str | Path,
    output_prefix: str | Path,
    output_format: str = "both",
) -> list[Path]:
    """Render SVG, PNG, or both and return generated paths."""
    graph = load_graph(input_path)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    formats = ("svg", "png") if output_format == "both" else (output_format,)
    outputs: list[Path] = []

    for fmt in formats:
        dot = build_graph(graph, output_format=fmt)
        outputs.append(Path(dot.render(filename=str(prefix), format=fmt, cleanup=True)))

    print(f"Graph nodes: {len(graph['nodes']):,}")
    print(f"Graph edges: {len(graph['edges']):,}")
    for output in outputs:
        print(f"Rendered: {output}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="F:/TrippinnI-work/outputs/reports/context/healthcare_context.json",
        help="Path to healthcare_context.json or bare graph JSON",
    )
    parser.add_argument(
        "--output-prefix",
        default="F:/TrippinnI-work/outputs/reports/visualizations/healthcare_knowledge_graph",
        help="Output path without the .svg/.png extension",
    )
    parser.add_argument("--format", choices=("svg", "png", "both"), default="both")
    args = parser.parse_args()
    render_graph(args.input, args.output_prefix, args.format)


if __name__ == "__main__":
    main()
