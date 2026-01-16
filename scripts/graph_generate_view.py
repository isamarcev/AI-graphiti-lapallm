import sys
from pathlib import Path
repo_root = Path("../graphity_lapa")

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from agent.graph import get_agent_app

agent = get_agent_app()

# 1. Mermaid diagram
mermaid = agent.get_graph().draw_mermaid()
with open("scripts/graph.mmd", "w") as f:
    f.write(mermaid)
print("✓ Saved to graph.mmd")

# 2. ASCII
print("\n" + agent.get_graph().draw_ascii())