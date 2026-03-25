import matplotlib.pyplot as plt
import networkx as nx

from graph_functions import CascadingFailureProcess


def draw_state(ax, graph: nx.Graph, pos: dict, active: dict, title: str) -> None:
    colors = ["#2a9d8f" if active[node] else "#d1495b" for node in graph.nodes()]
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#aaaaaa", width=0.7, alpha=0.6)
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=colors,
        node_size=80,
        linewidths=0.3,
        edgecolors="#333333",
    )
    ax.set_title(title)
    ax.set_axis_off()


def main() -> None:
    process = CascadingFailureProcess(
        n=50,
        m=2,
        alpha=0.2,
        seed=42,
        max_steps=50,
        redistribution_mode="capacity_weighted",
    )

    obs0 = process.reset(p_fail=0.1)
    r1 = process.step()
    obs1 = r1.observation
    r2 = process.step()
    obs2 = r2.observation

    graph = process.graph
    pos = nx.spring_layout(graph, seed=42)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    draw_state(axes[0], graph, pos, obs0["active"], "t=0")
    draw_state(axes[1], graph, pos, obs1["active"], "t=1")
    draw_state(axes[2], graph, pos, obs2["active"], "t=2")

    active0 = sum(obs0["active"].values())
    active1 = sum(obs1["active"].values())
    active2 = sum(obs2["active"].values())
    fig.suptitle(
        f"50-node BA Cascade (green=active, red=failed/inactive)\n"
        f"active counts: t0={active0}, t1={active1}, t2={active2}"
    )
    fig.tight_layout()
    fig.savefig("cascade_50_nodes_t0_t2.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
