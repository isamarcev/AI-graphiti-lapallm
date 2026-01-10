from agent.graph import get_agent_app

agent = get_agent_app()

# 1. Mermaid diagram
mermaid = agent.get_graph().draw_mermaid()
with open("graph.mmd", "w") as f:
    f.write(mermaid)
print("✓ Saved to graph.mmd")

# 2. ASCII
print("\n" + agent.get_graph().draw_ascii())