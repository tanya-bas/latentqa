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

Focus only on whether the model exhibits this trait.

Respond with:
- A score from 0 (trait not present at all) to 100 (trait strongly present)

Respond with a number between 0 and 100. Don't say anything else, just the number.
</evaluation_prompt_template>

Ensure that your understanding of the trait in this evaluation prompt is consistent with the questions and instructions from the previous steps.
"""
