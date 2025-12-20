# LlmPromptCategorization
Our hypothesis is that LLMs have parallel circuits for prompt categorization and response generation. We call this "Decoupled Intelligence".

## Hypotheses
This library investigates our hypotheses:
- **Structural Separation**: An LLM contains functionally distinct sub-networks (circuits) - Categorization Circuits, which map a prompt's intent to a task space, and Generation Circuits, which execute the specific logic of that task.
- **The Gating Mechanism**: Output selection is a "winner-takes-all" process where the activation magnitude of a Categorization Circuit acts as a gate, promoting the output of its paired Generation Circuit while suppressing others.
- **Hallucination Diagnostic Signal**: Model hallucination is not necessarily a failure of the generation logic, but a failure of the "Categorization Layer" to reach a clear consensus (low activation or multi-circuit interference)

More detail here:
https://docs.google.com/document/d/1x7n2iy1_LZXZNLQpxCzF84lZ8BEG6ZT3KWXC59erhJA 

## Find Good Models
The notebook CatGen_FindModels.ipynb: 
- Creates and saves synthetic maths data to synthetic_arithmetic_data.csv
- Finds good models that succeed on N tasks with M test examples and saves model names to GoodOpenModels_6Tasks_5Tests.json

## Investigate Good Models
