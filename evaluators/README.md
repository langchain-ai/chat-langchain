# Online Evaluators

`trajectory_accuracy.json` is the versioned contract for the LangSmith online run rule that emits the `trajectory_accuracy` feedback key. Deploy its `variable_mapping`, prompt, and structured-output `schema` together, and keep this file synchronized with the deployed rule.

The question mapping intentionally receives the root run's `inputs.messages`; the judge extracts the text after `User query:` so a child guardrail classifier prompt is not mistaken for the user's question. A missing final answer is scored as zero with reasoning rather than treated as an evaluator error.
