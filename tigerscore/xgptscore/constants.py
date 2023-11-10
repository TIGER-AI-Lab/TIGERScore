EVAL_ASPECTS = {}

# EVAL_ASPECTS["summarization"] = {
#     "Relevance": "How well the summary captures the **key** points of the input text. Consider whether all and **only** the important aspects are contained in the summary, not all the details.",
#     "Redundancy": "Whether the summary contains redundant information. Consider whether the summary contains information that is not necessary to understand the input text.",
#     "Consistency": "Whether the facts in the summary are consistent with the facts in the original article. Consider whether the summary does reproduce all facts accurately and does not make up untrue information.",
#     "Informativeness": "How well the summary is informative. Consider whether the summary is helpful for humans to understand the input text.",
#     "Fluency": "How well the summary is written in terms of grammar, spelling, punctuation, and coherence.",
#     # "Novelty": "The degree to which the summary presents new information or insights not explicitly stated in the original text. This aspect evaluates how well the summary provides new and valuable information not already conveyed in the source."
# }

# EVAL_ASPECTS["summarization"] = {
# "Relevance": "How well the summary captures the key points of the input text.",
# "Consistency": "Whether the facts in the summary are consistent with the facts in the input text.",
# # "Consistency": "Whether the facts in the summary are consistent with the facts in the input text.",
# # "Informativeness": "How well the summary is informative. Consider whether the summary is helpful for humans to understand the input text.",
# # "Coherence": "The quality of all sentences collectively, to the fit together and sound naturally.",
# "Fluency": "The quality of individual sentences, are they well-written and grammatically correct.",
# # "Redundancy": "Whether the summary contains redundant information. Consider whether the summary contains information that is not necessary to understand the input text.",
# }

# EVAL_ASPECTS["summarization"] = {
# "Relevance": "How well the summary captures the key points of the input text.",
# # "Fact Consistency": "Whether the facts in the summary are consistent with the facts in the input text.",
# "Consistency": "The factual alignment between the summary and the summarized source.",
# # "Informativeness": "How well the summary is informative. Consider whether the summary is helpful for humans to understand the input text.",
# "Coherence": "The collective quality of all sentences.",
# "Fluency": "The quality of individual sentences.",
# # "Redundancy": "Whether the summary contains redundant information. Consider whether the summary contains information that is not necessary to understand the input text.",
# }

EVAL_ASPECTS["summarization"] = {
    "Relevance": "This aspect refers to the degree to which the summarized output accurately reflects the key points of the input text. It assesses if the summary contains the necessary information from the original text. Example error types include: (1) Including irrelevant information, (2) Omitting important information, (3) Overemphasis on less important details, (4) Under-representation of key points, (5) Misrepresentation of main ideas, etc.",
    "Fact Consistency": "This aspect evaluates if the facts in the summary are consistent with the facts in the original text. It checks the accuracy of information presented in the summary. Example error types include: (1) Factual inaccuracies, (2) Misinterpretation, (3) Addition of non-existent facts, (4) Omission of factual information, etc.",
    "Coherence": "This aspect pertains to the logical and meaningful arrangement of ideas in the summary. It checks if the summary makes sense as a standalone text. Example error types include: (1) Illogical sequence of ideas, (2) Lack of clear connections between ideas, (3) Poor paraphrasing, (4) Lack of thematic unity, etc.",
    "Fluency": "This aspect reviews the model-generated output's use of language, including grammar, punctuation, and vocabulary that affect the quality of the sentences. Example error types include: (1) Spelling mistakes, (2) Grammatical errors, (3) Vocabulary misuse, (4) Style and tone inconsistencies, (5) Garbled characters, etc.",
}

# EVAL_ASPECTS["translation"] = {
# #    # "Missing Translation": "Measure by words that are not translated in the output.",
# #     "Extra Translation": "Measure by words that are translated but not in the input.",
# # "Semantic Shift": "Measured by the extent to which the meaning of a sentence or phrase changes during the translation process despite all words being translated correctly.",
#     "Error Translation": "Measure by words that are translated incorrectly in the output, or words that are not translated in the output, or words that are translated but not in the input.",
#     "Incoherent": "Measured by words that are translated **correctly** but the concatenated output is not a coherent sentence in the target language.",
#     "Style Matching": "Measured by the extent to which the output is consistent with the desired style or genre of the user or the task.",
# }
# EVAL_ASPECTS["translation"] = {
#     "Accuracy": "Measured by the errors where words that additionally, inaccurately, or not translated in the output.",
#     "Fluency": "Measured by the errors in spelling, grammar, punctuation, register, inconsistency, and character encoding, etc. in the output.",
#     "Terminology": "Measured by the errors in terminology, such as the use of proper nouns, technical terms, and abbreviations, etc. And the locale conventions, such as date, time, currency, and number formats, etc.",
#     "Style Matching": "Measured by the extent to which the output is consistent with the desired style or genre of the user or the task.",
# }
# EVAL_ASPECTS["translation"] = {
#     "Accuracy": "Measured by identifying errors in the translation, such as addition of information not present in the source, omission of content from the source, mistranslation that inaccurately represents the source, and untranslated text where portions of the source text are not translated in the output.",
#     "Fluency": "Measured by the errors in translation, including spelling mistakes, grammar issues (excluding orthography), incorrect register, internal inconsistencies (unrelated to terminology), and garbled characters from incorrect encoding",
#     "Terminology": "Measured by the errors in terminology, such as the use of proper nouns, technical terms, and abbreviations, etc. And the locale conventions, such as date, time, currency, and number formats, etc.",
#     "Style Matching": "Measured by the extent to which the output is consistent with the desired style or genre of the user or the task.",
# }
EVAL_ASPECTS["translation"] = {
    "Accuracy": "This aspect refers to the degree to which the translated text adheres to the original text, maintaining the same meaning, context and cultural nuances. Example error types include: (1) Literal translation, (2) Omission, (3) Addition, (4) Mistranslation, (5) False friends , etc.",
    "Fluency": "This aspect refers to how naturally the translation reads in the target language. It should not sound like a translation but like a text originally written in the target language. Example error types include: (1) Grammar mistakes, (2) Syntax errors, (3) Inappropriate language registe, (4) Use of unnatural expressions, (5) Inconsistent tense usage, etc.",
    "Terminology": "This aspect refers to the appropriate use of specific terms and jargon related to a particular field or industry. Example error types include: (1) Incorrect use of proper nouns, (2) Incorrect use of technical terms, (3) Incorrect use of abbreviations, (4) Incorrect date, time, currency, and number formats, etc.",
    "Style Matching": "This aspect refers to the translator's ability to maintain the same style, tone, and voice as the original text. Example error types include: (1) Inconsistent style, (2) Inconsistent tone, (3) Inconsistent voice, etc.",
}

# EVAL_ASPECTS["data2text"] = {
#     # "Fact Completeness": "Measure by the percentage of specified facts that are adequately captured in the generated sentences.",
#     # "Non-Redundancy": "Measure by the percentage of generated sentences that are not redundant.",
#     # May we should give more aspects for data2text
#     #  "Information Completeness": "Measured by the percentage of generated answers that include all the relevant information provided in the input text.",
#     "Relevance": "How well the output captures the key points of the input text.",
#     # "Logical Coherence": "Measure by the percentage of generated sentences that are coherent and logical.",
#     "Fluency": "The quality of sentences.",
# }
EVAL_ASPECTS["data2text"] = {
    "Accuracy": "This aspect deals with the correctness of the information presented by the output. It checks if the output is factually accurate and matches with the information provided in the input. Example error types include: (1) Factual inaccuracies, (2) Misinterpretation, (3) Inaccurate Data Conversion, (4) Contradictory information, (5) Incorrect reasoning, etc.",
    "Logical Coherence": "This aspect evaluates how well the output transforms structured data into a comprehensible, logical, and engaging text. Example error types include: (1) Illogical sequence, (2) Inconsistent facts, (3) Poor paraphrasing, (4) Inadequate explanation, (5) Overcomplication, etc.",
    "Fluency": "This aspect reviews the model-generated output's use of language, including grammar, punctuation, and vocabulary that affect the quality of the sentences. Example error types include: (1) Spelling mistakes, (2) Grammatical errors, (3) Vocabulary misuse, (4) Style and tone inconsistencies, (5) Garbled characters, etc.",
}

EVAL_ASPECTS["mathQA"] = {
    "Problem Understanding": "This aspect assesses how well the output accurately comprehend the text-based description of the math problem. Example error types include: (1) Problem misinterpretation, (2) Neglect of information, (3) Incorrect identification of variables, (4) Overinterpretation (5) Incorrect Assumptions, etc.",
    "Problem Formulation": "This aspect involves translating the problem from a textual form into a mathematical equation or set of equations that can be solved. Example error types include: (1) Incorrect formula usage, (2) Incorrect variable usage, (3) Incorrect operator usage, (4) Missing steps, (5) Misalignment with problem, etc.",
    "Computing Accuracy": "This aspect assesses the output's ability to perform the mathematical operations accurately to arrive at the correct solution. Example error types include: (1) Arithmetic errors, (2) Incorrect variable assignment, (3) Incorrect operation order, (4) Misuse of operation rules, etc.",
    "Solution Interpretation": "This involves the how well the output correctly interpret the solution of the problem in the context of the original problem statement. Example error types include: (1) Incorrect contextual interpretation, (2) Ignoring units of measurement (3) Not addressing all parts of the problem, etc.",
}

# EVAL_ASPECTS["long-form QA"] = {
#     "Fact Accuracy": "Measured by the percentage of generated answers that accurately reflect the facts retrieved from the knowledge source.",
#     # "Fact Missing": "Measured by the percentage of generated answers that are missing some facts retrieved from the knowledge source.",
#     "Specific": "Measured by the percentage of generated answers that are detailed, specific, and step-by-step.",
#     "Coherence": "Measured by the percentage of generated answers that are coherent and logical.",
#     "Informativeness": "Measured by the percentage of generated answers that are helpful in answering the question.",
# }
EVAL_ASPECTS["long-form QA"] = {
    "Accuracy": "This aspect evaluates the factual correctness of the answer. It checks if the information provided in the response is true and matches with the facts available from reliable sources. Example error types include: (1) Factual inaccuracies, (2) Misinterpretation, (3) Outdated information, (4) Contradictory information, (5) Incorrect inference, etc.",
    "Completeness": "This aspect checks if the response provided answers all parts of the question comprehensively. It evaluates if the answer leaves out any critical parts or details that were asked in the question. Example error types include: (1) Partial answer, (2) Omission of key details, (3) Lack of context, (4) Lack of depth, (5) Ignoring multi-part questions, etc.",
    "Informativeness": "This aspect assesses the quality of the response in terms of how helpful it is for the user to understand the answer. It checks if the response is clear, concise, and easy to understand. Example error types include: (1) Redundancy, (2) Irrelevance, (3) Overuse of jargon, (4) Lack of examples, (5) Lack of sources, etc.",
    "Clarity": "This aspect assesses the readability and understandability of the response. Errors in this stage could lead to answers that are difficult to understand, despite being factually correct and based on accurate information synthesis. Example error types include: (1) Unclear language, (2) Poor organization, (3) Lengthy responses, (4) Unexplained jargon, (5) Grammar or spelling errors, etc.",
}
# EVAL_ASPECTS["long-form QA"] = {
#     "Accuracy": "This aspect evaluates the factual correctness of the answer. It checks if the information provided in the response is true and matches with the facts available from reliable sources. Example error types include: (1) Factual inaccuracies, (2) Misinterpretation, (3) Outdated information, (4) Contradictory information, (5) Incorrect inference, etc.",
#     # "Completeness": "This aspect checks if the response provided answers all parts of the question comprehensively. It evaluates if the answer leaves out any critical parts or details that were asked in the question. Example error types include: (1) Partial answer, (2) Omission of key details, (3) Lack of context, (4) Lack of depth, (5) Ignoring multi-part questions, etc.",
#     # "Informativeness": "This aspect assesses the quality of the response in terms of how helpful it is for the user to understand the answer. It checks if the response is clear, concise, and easy to understand. Example error types include: (1) Redundancy, (2) Irrelevance, (3) Overuse of jargon, (4) Lack of examples, (5) Lack of sources, etc.",
#     "Clarity":"",
#     "Fluency": "This aspect assesses the readability and understandability of the response. Errors in this stage could lead to answers that are difficult to understand, despite being factually correct and based on accurate information synthesis. Example error types include: (1) Unclear language, (2) Poor organization, (3) Lengthy responses, (4) Unexplained jargon, (5) Grammar or spelling errors, etc.",
# }
EVAL_ASPECTS["instruction-following"] = {
    "Comprehension": "This aspect evaluates how well the output understands the given instruction. Example error types include: (1) Misinterpretation, (2) Inability to grasp complex instructions, (3) Failure to understand context, etc.",
    "Accuracy": "This aspect measures the correctness of the output in relation to the instruction and the paired input context. Example error types include: (1) Inconsistent content with instruction and input, (2) Factual errors, (3) Misapplication of instructions, (4) Incorrect reasoning, etc.",
    "Informativeness": "This aspect assesses the relevancy and usefulness of the information provided by the output. Example error types include: (1) Irrelevant information, (2) Insufficient information, (3) Overload of information, (4) Outdated information, (5) Repetitive information, etc.",
    "Coherence": "This aspect evaluates how logically the output flows and connects. Errors in this stage could lead to answers that are difficult to understand, despite being factually correct and based on accurate information synthesis. Example error types include: (1) Grammar or spelling errors, (2) Poor organization, (3) Illogical sequence, (4) Inappropriate tone or style, (5) Non-sequiturs, etc.",
}

EVAL_ASPECTS["story_generation"] = {
    "Fluency": "How well the story is written in terms of grammar, spelling, punctuation, and coherence.",
    "Consistency": "How well the story is consistent in terms of plot, characters, setting, and themes.",
    "Style Matching": "How well the story matches the desired style or genre of the user or the task.",
    # "Interestingness": "How interesting the story is in terms of plot, characters, setting, and themes.", # maybe subjective
    # "Novelty": "How original and diverse the story is in terms of plot, characters, setting, and themes." # maybe delete
}

EVAL_ASPECTS["other"] = {
    "Consistency": "How well the generated text is consistent the input. It could be the consistency of the style, the topic, the genre, etc.",
    "Informativeness": "How well the generated text is informative.",
    "Rationality": "How well the generated text is rational. It should be consistent with the common sense.",
}

EVAL_ASPECTS["common_sense"] = {
    "Fact Accuracy": "Measured by the percentage of generated text that correctly states the facts,",
    "Relevance": "Measured by the percentage of generated text that is relevant to the input, such as concepts, entities, and relations",
    "Informativeness": "Measure by the percentage of generated facts in sentences that are helpful for humans to understand the data.",
}  # delay

EVAL_ASPECTS["image_captioning"] = {
    "Objects Accuracy": "The percentage of objects and their attributes in the image that are correctly mentioned in the caption.",
    "Actions Accuracy": "The percentage of actions performed by the objects and and relations between objects in the image that are correctly mentioned in the caption.",
    "Detail Richness": "Measured by the richness of the specific details for both the objects and actions in the image that are correctly mentioned in the caption.",
}  # not do this for now
