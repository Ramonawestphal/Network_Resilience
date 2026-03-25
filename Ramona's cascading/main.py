from graph_functions import CascadingFailureProcess, plot_cascade_history


def run_cascade_demo() -> None:
    process = CascadingFailureProcess(
        n=30,
        m=2,
        alpha=0.25,
        seed=42,
        max_steps=20,
        redistribution_mode="capacity_weighted",
    )

    obs0 = process.reset(p_fail=0.10)
    history = [{"observation": obs0.copy(), "new_failures": obs0.get("new_failures", []).copy()}]

    print(f"t=00 initial_failures={len(obs0['new_failures'])} active={sum(obs0['active'].values())}/{process.n}")

    while not process.done:
        result = process.step()
        info = result.info

        history.append(
            {
                "observation": result.observation.copy(),
                "new_failures": info["new_failures"].copy(),
            }
        )

        print(
            f"t={info['t']:02d} processed={len(info['processed_failures'])} "
            f"new_failures={len(info['new_failures'])} "
            f"active={info['active_count']}/{process.n}"
        )

        if result.done:
            break

    plot_cascade_history(process.graph, history, cols=4)


if __name__ == "__main__":
    run_cascade_demo()
