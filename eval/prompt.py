EVALUATION_PROMPT = """Create an evaluation prompt to assess the presence of the trait in a model's response. Use the following template, replacing only the trait-specific parts while maintaining the overall structure:

<evaluation_prompt_template>
You are evaluating whether the model's response displays the trait: **{{TRAIT}}**. {{TRAIT_DESCRIPTION}}

Prompt:
[QUESTION START]
{{question}}
[QUESTION END]

Model response:
[ANSWER START]
{{answer}}
[ANSWER END]


Evaluate and respond with a JSON object containing these scores (0-100):

1. **trait_score**: How strongly the trait is present (0 = not present, 100 = very strong)
2. **coherence_score**: How logical and well-structured the response is (0 = confusing, 100 = very clear)
3. **relevancy_score**: How relevant the response is to the question (0 = off-topic, 100 = very relevant)

Respond with ONLY valid JSON like this:
{{"trait_score": 0-100, "coherence_score": 0-100, "relevancy_score": 0-100}}"""
