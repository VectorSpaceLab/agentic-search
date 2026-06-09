DatasetPrompts = {
    # StackExchange
    "biology": (
        "biology post",
        "passage",
        "Given a query (biology post) and a document (passage), the document is relevant to the query if the critical concepts or theories discussed in the document can provide references for domain experts to draft an answer to the query."
    ),
    "earth_science": (
        "earth science post",
        "passage",
        "Given a query (earth science post) and a document (passage), the document is relevant to the query if the critical concepts or theories discussed in the document can provide references for domain experts to draft an answer to the query."
    ),
    "economics": (
        "economics post",
        "passage",
        "Given a query (economics post) and a document (passage), the document is relevant to the query if the critical concepts or theories discussed in the document can provide references for domain experts to draft an answer to the query."
    ),
    "psychology": (
        "psychology post",
        "passage",
        "Given a query (psychology post) and a document (passage), the document is relevant to the query if the critical concepts or theories discussed in the document can provide references for domain experts to draft an answer to the query."
    ),
    "robotics": (
        "robotics post",
        "passage",
        "Given a query (robotics post) and a document (passage), the document is relevant to the query if the critical concepts or theories discussed in the document can provide references for domain experts to draft an answer to the query."
    ),
    "stackoverflow": (
        "Stack Overflow post",
        "passage",
        "Given a query (Stack Overflow post) and a document (passage), the document is relevant to the query if the critical concepts or theories discussed in the document can provide references for domain experts to draft an answer to the query."
    ),
    "sustainable_living": (
        "sustainable living post",
        "passage",
        "Given a query (sustainable living post) and a document (passage), the document is relevant to the query if the critical concepts or theories discussed in the document can provide references for domain experts to draft an answer to the query."
    ),

    # Coding
    "leetcode": (
        "LeetCode problem",
        "coding problem solution",
        "Given a query (LeetCode problem) and a document (coding problem solution), the document is relevant to the query if the underlying algorithms or data structures used in the document can provide helpful insights for solving the problem in the query."
    ),
    "pony": (
        "Pony coding instruction",
        "Pony documentation passage",
        "Given a query (Pony coding instruction) and a document (Pony documentation passage), the document is relevant to the query if the Pony syntax described in the document is necessary for beginners with no prior knowledge of Pony to complete the coding instruction in the query."
    ),

    # Theorem-based
    "aops": (
        "math problem",
        "math problem solution",
        "Given a query (math problem) and a document (math problem solution), the document is relevant to the query if the theorems used in the document can provide helpful insights for solving the problem in the query."
    ),
    "theoremqa_questions": (
        "math problem",
        "math problem solution",
        "Given a query (math problem) and a document (math problem solution), the document is relevant to the query if the theorems used in the document can provide helpful insights for solving the problem in the query."
    ),
    "theoremqa_theorems": (
        "math problem",
        "math-related passage",
        "Given a query (math problem) and a document (math-related passage), the document is relevant to the query if the theorem described in the document can help solve the problem in the query."
    ),
}


PromptTemplate = """\
Here is the **relevance definition** in a retrieval task: {relevance_definition}

Now given a **query** ({query_type}) and a **document** ({doc_type}) in this retrieval task, your mission is to perform the following steps to determine the relevance between the query and the document.

1. Query Analysis: Think to reason and describe what information would be most helpful in answering the query.
2. Document Analysis: Discuss how the information provided by the document fulfills or fails to fulfill the requirements implied by the query.
3. Relevance Annotation: Based on the relevance definition and the insights from the previous two steps, clearly justify your final relevance annotation result and annotate an integer score from a scale of 1 to 5. Please use the following guide:
    - **5 (Highly Relevant):** The document is directly and fully responsive to the query, providing comprehensive, accurate, and specific information that completely addresses all aspects of the query.
    - **4 (Relevant):** The document is largely relevant and provides most of the information needed, but may have minor omissions, slight inaccuracies, or not be perfectly aligned with the query's intent.
    - **3 (Moderately Relevant):** The document has some relevance and offers partial information, but it may be incomplete, vague, or include some irrelevant content. It provides a basic connection but lacks depth or precision.
    - **2 (Slightly Relevant):** The document has minimal relevance, with only a small portion of content tangentially related to the query. The majority of the document is off-topic or provides little value.
    - **1 (Irrelevant):** The document is completely unrelated to the query and provides no useful information. There is no discernible connection or value for answering the query.

After providing your detailed analysis and justification for all the steps above, conclude your entire response with the final relevance score. The score must be placed strictly between the <score> tags. There should be no other text or explanation inside the tags:
<score>
[From a scale of 1 to 5, annotate the degree of relevance between the query and the document.]
</score>

Note: The whole response should be as concise as possible while covering all the necessary details, and not exceeding 512 words in total.

Query ({query_type}):
[Begin of Query]
{query}
[End of Query]

Document ({doc_type}):
[Begin of Document]
{doc}
[End of Document]
"""
