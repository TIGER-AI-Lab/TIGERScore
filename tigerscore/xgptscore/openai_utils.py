import os
if os.environ.get('OPENAI_API_TYPE', None) == 'azure':
    # pip install openai<=0.28.1, fire, numpy, tiktoken
    from .openai_utils_azure import (
        openai_completions,
        _prompt_to_chatml,
        _chatml_to_prompt,
    )
    import openai
    assert openai.VERSION <= "0.28.1", "Azure API is only supported in openai-python 0.28.1 or later."
elif os.environ.get('OPENAI_UTILS_TYPE', None) == 'curl':
    # pip install openai>=1.0.0, fire, numpy, tiktoken
    from .openai_utils_curl import (
        openai_completions,
        _prompt_to_chatml,
        _chatml_to_prompt,
    )
    import openai
    assert openai.VERSION >= "1.0.0", "OpenAI API is only supported in openai-python 1.0.0 or later."
else:
    # pip install openai>=1.0.0, fire, numpy, tiktoken
    from .openai_utils_openAI import (
        openai_completions,
        _prompt_to_chatml,
        _chatml_to_prompt,
    )
    import openai
    assert openai.VERSION >= "1.0.0", "OpenAI API is only supported in openai-python 1.0.0 or later."