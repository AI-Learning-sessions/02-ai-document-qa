from src.vector_store import get_collection
from src.retriever import retrieve_relevant_chunks
from src.llm import create_client, generate_answer


TEST_CASES = [
    {
        "question": "What is the OSI model?",
        "expected_keywords": [
            "seven layers",
            "physical",
            "data link",
            "network",
            "transport",
            "session",
            "presentation",
            "application"
        ]
    },
    {
        "question": "What is SNMP?",
        "expected_keywords": [
            "monitor",
            "manage",
            "network devices"
        ]
    },
    {
        "question": "What does SNMPv3 provide?",
        "expected_keywords": [
            "authentication",
            "privacy"
        ]
    },
    {
        "question": "Who invented the telephone?",
        "expected_keywords": []
    }
]


collection = get_collection()

try:
    llm_client = create_client()
except ValueError as e:
    print(e)
    exit()


for test_case in TEST_CASES:

    question = test_case["question"]
    expected_keywords = test_case["expected_keywords"]

    print("\n" + "=" * 60)
    print(f"Question: {question}")
    print("=" * 60)

    documents, metadatas, distances = (
        retrieve_relevant_chunks(
            collection,
            question
        )
    )

    if not documents:

        print("No relevant chunks found.")

        if not expected_keywords:
            print("Result: PASS")
        else:
            print("Result: FAIL")

        continue

    context = "\n\n".join(documents)

    try:

        answer = generate_answer(
            llm_client,
            question,
            context
        )

        print("\nAnswer:")
        print(answer)

        answer_lower = answer.lower()

        if not expected_keywords:

            print(
                "\nExpected: "
                "No answer from the document"
            )

        else:

            missing_keywords = []

            for keyword in expected_keywords:

                if keyword.lower() not in answer_lower:
                    missing_keywords.append(keyword)

            if not missing_keywords:

                print("\nResult: PASS")

            else:

                print("\nResult: REVIEW")

                print(
                    "Missing keywords:",
                    missing_keywords
                )

    except Exception as e:

        print("\nGemini API error:")
        print(e)