# Knowledge graph visualization

TrippinnI stores the semantic graph in
`outputs/reports/context/healthcare_context.json` under the
`knowledge_graph` key. Render it with Graphviz:

1. Install Python dependencies with `pip install -r requirements.txt`.
2. Install the Graphviz system executable and ensure `dot` is on PATH.
   On Windows, install Graphviz for Windows and reopen the terminal.
3. Run from the repository root:

```powershell
python -m visualization.knowledge_graph_viz
```

By default this writes SVG and PNG to
`F:/TrippinnI-work/outputs/reports/visualizations/healthcare_knowledge_graph.*`.
Override paths or format as needed:

```powershell
python -m visualization.knowledge_graph_viz --input path/to/healthcare_context.json --output-prefix outputs/reports/visualizations/healthcare_graph --format svg
```

SVG is recommended for large graphs because it remains sharp when zoomed.
The renderer accepts the wrapped context report or a bare graph payload. It
adds display-only placeholders for edge endpoints that the graph builder has
not explicitly materialized as nodes; it does not modify the source JSON.
