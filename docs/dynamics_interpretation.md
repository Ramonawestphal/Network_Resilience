# Adopted Dynamics Interpretation

This project uses the following round structure for cascading failure and recovery.

## Exogenous Failure

- There is exactly one exogenous failure event at `t = 0`.
- During that event, multiple nodes may fail independently with probability `pfail`.

## Cascade Waves

- Failed nodes redistribute load only to their direct active neighbors.
- This redistribution is local, but cascade effects can still spread over multiple hops over time.
- A cascade wave only redistributes load from the current failed frontier.
- Nodes that overload during that wave become the failed frontier for the next cascade wave.

## Recovery Rounds

- In each recovery round, the agent may reactivate up to `B` failed nodes.
- These `B` repairs are modeled as sequential single-node decisions.
- Repaired nodes re-enter with their original capacity and zero load.

## Round Order

The adopted step order is:

1. initial random failures at `t = 0`
2. repair up to `B` failed nodes
3. run one cascade wave from the current failed frontier
4. repair up to `B` failed nodes
5. run the next cascade wave
6. repeat until there are no failed nodes left or the episode horizon is reached

This means there is only one exogenous failure round, but potentially many endogenous cascade waves.

## Reward Semantics

- Each repair action is rewarded by the ANC improvement immediately after that repair.
- Intermediate repairs inside a round do not trigger a cascade.
- When a repair exhausts the round budget, the cascade wave happens after the reward is computed and affects the next observation rather than the current action reward.
