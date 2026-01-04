## 🧠 What is Deep Research?

**Deep Research** refers to an automated research paradigm that goes beyond simple search or question-answering.
It aims to replicate structured human research workflows by combining:

* **Explicit research planning**
* **Iterative and targeted information retrieval**
* **Multi-source synthesis**
* **Structured report generation**
* **(Optionally) evaluation and reflection loops**

Instead of producing a single response, deep research systems generate **comprehensive, well-organized research outputs**, often resembling professional reports, literature reviews, or analytical briefs.

This repository serves as a practical lab for understanding how different architectural choices impact the quality, speed, and controllability of deep research systems.

---

## 🎯 Purpose of This Repository

This project was created to:

* Study **multiple deep research architectures** side by side
* Understand trade-offs between **control, speed, and autonomy**
* Experiment with **workflow-based vs. multi-agent designs**
* Build and iterate on **custom deep research systems**
* Serve as a foundation for future **research tools or services**

It is intentionally structured as an **experimental and comparative lab**, rather than a single monolithic implementation.

---

## 📚 Reference & Inspiration

This repository is inspired by and builds upon the open-source
**[Open Deep Research](https://github.com/langchain-ai/open_deep_research)** project developed by **LangChain**, which is licensed under the MIT License.

The LangChain Open Deep Research project provides:

* A production-ready deep research agent
* Multiple architectural approaches (workflow, multi-agent, modern LangGraph-based)
* Extensive configuration for models, search tools, and MCP servers
* Evaluation pipelines aligned with Deep Research Bench

The implementations and ideas in this repository were developed by **studying, re-implementing, and extending** those concepts for learning, experimentation, and architectural exploration.

👉 For the original and production-grade implementation, please refer to the official LangChain repository.

---

## 🏗️ Repository Structure

```text
.
├─ upstream/                 # Reference implementations (from Open Deep Research)
│  ├─ legacy_graph/           # Workflow-based deep research
│  ├─ legacy_multi_agent/     # Multi-agent deep research
│  └─ latest/                 # Modern deep research implementation
│
├─ my_research/               # Custom and experimental implementations
│  ├─ hybrid/                 # Hybrid workflow + multi-agent designs
│  ├─ prompts/                # Prompt engineering experiments
│  └─ evaluators/             # Quality and evaluation logic
│
├─ docs/                      # Design notes and architectural analysis
└─ README.md
```

---

## 🏛️ Referenced Implementations

### 1️⃣ Workflow-Based Deep Research

* Explicit **plan → execute → reflect** structure
* Sequential section generation
* Human-in-the-loop friendly
* High controllability and traceability

### 2️⃣ Multi-Agent Deep Research

* Supervisor + researcher agent architecture
* Parallel research execution
* Faster report generation
* Useful for exploratory or time-sensitive research

### 3️⃣ Modern Deep Research Systems

* State-driven orchestration
* Structured outputs
* Evaluation and scoring loops
* Designed for scalability and reuse

---

## 🔬 Custom Experiments

In addition to referenced implementations, this repository includes:

* Hybrid architectures combining **workflow control with agent parallelism**
* Custom planning and evaluation strategies
* Prompt design experiments
* Research quality assessment logic

These components represent **original work and ongoing experimentation**.

---

## ⚖️ License

This project is licensed under the **MIT License**.

It includes and builds upon code from the
**Open Deep Research** project by LangChain (MIT License).

Original work:

* Copyright (c) 2025 LangChain

Additional modifications and original implementations:

* Copyright (c) 2026 GEUNHONGMIN

See the [LICENSE](./LICENSE) file for full details.

---

## 🚧 Disclaimer

This repository is intended for **research, learning, and experimentation**.
It is **not** a drop-in replacement for the official LangChain Open Deep Research system.

For production usage, benchmarks, and maintained implementations, please refer to the original project.

