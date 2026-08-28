BL_FULL_PROMPT ="""Write a bulleted list of feedback for the following text:

{story}

The feedback should state whether the text is:
- not fluent (e.g. typos, grammar issues)
- not coherent (lacking internally consistency or badly ordered)
- lacking clarity (e.g. text is wordy or ambiguous)
- needs a stylistic change (e.g. formality, tone, other writing preferences)
- needs additions (e.g. needs more information or updated information)
- needs something removed (e.g. get rid of something that doesn’t fit with the rest of the document)
If the story has no problems, say ‘The text is perfect as-is’.
Feedback:
"""

BL_ONLY_PROMPT ="""Write a bulleted list of feedback for the following text:

{story}

If the story has no problems, say ‘The text is perfect as-is’.
Feedback:
"""


def get_prompt_template(prompt_type: str) -> str:
    """Return the prompt template based on the prompt type."""
    if prompt_type == "bl_full":
        return BL_FULL_PROMPT
    elif prompt_type == "bl_only":
        return BL_ONLY_PROMPT
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")