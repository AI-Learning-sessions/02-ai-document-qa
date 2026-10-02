from src.vector_store import get_collection
from src.embeddings import create_embeddings


TEST_CASES = [
    {
        "question": "What is the OSI model?",
        "expected_page": 3
    },
    {
        "question": "What is SNMP?",
        "expected_page": 18
    },
    {
        "question": "What is TCP?",
        "expected_page": 8
    },
    {
        "question": "What is SSH?",
        "expected_page": 17
    },
    {
        "question": "What is ICMP?",
        "expected_page": 19
    },
    {
        "question": "What is a network protocol?",
        "expected_page": 2
    },
    {
        "question": "What is the TCP/IP model?",
        "expected_page": 3
    },
    {
        "question": "Who invented the telephone?",
        "expected_page": None
    }
]


collection = get_collection()

print("Starting retrieval evaluation...\n")

hit_at_1 = 0
hit_at_3 = 0

known_answer_questions = 0

negative_tests = 0
negative_tests_passed = 0


for test_case in TEST_CASES:

    question = test_case["question"]
    expected_page = test_case["expected_page"]

    query_embedding = create_embeddings(
        [question]
    )[0]

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=3
    )

    retrieved_pages = [
        metadata["page"]
        for metadata in results["metadatas"][0]
    ]

    distances = results["distances"][0]

    print(f"Question: {question}")
    print(f"Expected page: {expected_page}")
    print(f"Retrieved pages: {retrieved_pages}")

    # --------------------------------
    # Known-answer question
    # --------------------------------

    if expected_page is not None:

        known_answer_questions += 1

        if retrieved_pages[0] == expected_page:
            print("Hit@1: PASS")
            hit_at_1 += 1
        else:
            print("Hit@1: FAIL")

        if expected_page in retrieved_pages:
            print("Hit@3: PASS")
            hit_at_3 += 1
        else:
            print("Hit@3: FAIL")

    # --------------------------------
    # Negative question
    # --------------------------------

    else:

        negative_tests += 1

        print("Expected: No relevant document page")

        # Check whether all retrieved
        # results are above our threshold.

        if all(
            distance > 1.4
            for distance in distances
        ):
            print("Negative test: PASS")
            negative_tests_passed += 1
        else:
            print("Negative test: FAIL")

    print("\nDistances:")

    for i, distance in enumerate(distances):
        print(
            f"  Result {i + 1}: "
            f"{distance:.4f} "
            f"(Page {retrieved_pages[i]})"
        )

    print("-" * 50)


# ========================================
# Final evaluation summary
# ========================================

print("\n")
print("=" * 50)
print("EVALUATION SUMMARY")
print("=" * 50)

print(
    f"\nKnown-answer questions: "
    f"{known_answer_questions}"
)

print(
    f"Hit@1: "
    f"{hit_at_1}/{known_answer_questions}"
)

print(
    f"Hit@3: "
    f"{hit_at_3}/{known_answer_questions}"
)

print(
    f"\nNegative tests: "
    f"{negative_tests}"
)

print(
    f"Negative tests passed: "
    f"{negative_tests_passed}/{negative_tests}"
)

print("=" * 50)