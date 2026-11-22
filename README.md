# when-to-replan

**ErrandWorld** is a controlled testbed for studying when LLM agents benefit from explicit, revisable plans. It independently varies task length and plan-invalidating disruptions to separate the challenges of long tasks from those of changing environments. The project compares ReAct, adaptive replanning, and local repair across three models under matched token and action budgets, using a deterministic simulator and ground truth computed by a classical search planner. Its goal is to reveal when maintaining and revising a plan pays off—and when simpler strategies suffice.
