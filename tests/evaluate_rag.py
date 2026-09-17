
import sys
from pathlib import Path

from langsmith import Client


# ---------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# ---------------------------------------------------------
# Import chatbot service
# ---------------------------------------------------------

from src.services.chatbot_service import ChatbotService

from tests.evaluation_dataset import (
    EVALUATION_DATASET,
)


# =========================================================
# BASIC EVALUATION FUNCTIONS
# =========================================================


# ---------------------------------------------------------
# Check Answer
# ---------------------------------------------------------

def check_answer(
    actual_answer: str,
    expected_keywords: list[str],
) -> bool:
    """
    Check whether all expected keywords
    are present in the actual answer.
    """

    actual = actual_answer.lower()

    for keyword in expected_keywords:

        if keyword.lower() not in actual:
            return False

    return True


# ---------------------------------------------------------
# Check Sources
# ---------------------------------------------------------

def check_sources(
    actual_sources: list[str],
    expected_sources: list[str],
) -> bool:
    """
    Check whether at least one expected relevant source
    is present in the retrieved sources.
    """

    actual = set(actual_sources)
    expected = set(expected_sources)

    return bool(
        actual.intersection(expected)
    )


# =========================================================
# EXISTING RAG EVALUATION
# =========================================================


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_rag():

    print("\n")

    print("=" * 70)

    print(
        "              MEMORYFAQ-RAG ANSWER EVALUATION"
    )

    print("=" * 70)


    # -----------------------------------------------------
    # Create chatbot
    # -----------------------------------------------------

    print("\nInitializing chatbot...")

    chatbot = ChatbotService()


    # -----------------------------------------------------
    # Counters
    # -----------------------------------------------------

    total_questions = len(
        EVALUATION_DATASET
    )

    passed_questions = 0

    supported_questions = 0
    supported_passed = 0

    unsupported_questions = 0
    unsupported_rejected = 0


    # -----------------------------------------------------
    # Run every evaluation question
    # -----------------------------------------------------

    for index, test_case in enumerate(
        EVALUATION_DATASET,
        start=1,
    ):

        question = test_case[
            "question"
        ]

        expected_keywords = test_case[
            "expected_keywords"
        ]

        expected_sources = test_case[
            "relevant_sources"
        ]

        question_type = test_case[
            "type"
        ]


        print("\n")

        print("-" * 70)

        print(
            f"Question {index}"
        )

        print("-" * 70)

        print(
            f"Question: {question}"
        )

        print(
            f"Type: {question_type}"
        )


        # -------------------------------------------------
        # Ask chatbot
        # -------------------------------------------------

        try:

            result = chatbot.ask(
                question
            )

        except Exception as e:

            print(
                f"\n❌ ERROR: {e}"
            )

            print(
                "Result: ❌ FAIL"
            )

            continue


        # -------------------------------------------------
        # Get result
        # -------------------------------------------------

        actual_answer = result[
            "answer"
        ]

        actual_sources = result[
            "sources"
        ]


        print(
            "\nActual Answer:"
        )

        print(
            actual_answer
        )

        print(
            "\nActual Sources:"
        )

        print(
            actual_sources
        )


        # =================================================
        # SUPPORTED QUESTION
        # =================================================

        if question_type == "supported":

            supported_questions += 1


            # ---------------------------------------------
            # Answer check
            # ---------------------------------------------

            answer_pass = check_answer(
                actual_answer,
                expected_keywords,
            )


            # ---------------------------------------------
            # Source check
            # ---------------------------------------------

            source_pass = check_sources(
                actual_sources,
                expected_sources,
            )


            print("\n")

            print(
                f"Answer Check: "
                f"{'✅ PASS' if answer_pass else '❌ FAIL'}"
            )

            print(
                f"Source Check: "
                f"{'✅ PASS' if source_pass else '❌ FAIL'}"
            )


            # ---------------------------------------------
            # Final result
            # ---------------------------------------------

            if answer_pass and source_pass:

                print(
                    "\nResult: ✅ PASS"
                )

                passed_questions += 1
                supported_passed += 1

            else:

                print(
                    "\nResult: ❌ FAIL"
                )


        # =================================================
        # UNSUPPORTED QUESTION
        # =================================================

        elif question_type == "unsupported":

            unsupported_questions += 1


            # ---------------------------------------------
            # Unsupported question should return
            # no sources.
            # ---------------------------------------------

            rejected = (
                len(actual_sources) == 0
            )


            print("\n")

            print(
                f"Unsupported Rejection: "
                f"{'✅ PASS' if rejected else '❌ FAIL'}"
            )


            if rejected:

                print(
                    "\nResult: ✅ PASS"
                )

                passed_questions += 1
                unsupported_rejected += 1

            else:

                print(
                    "\nResult: ❌ FAIL"
                )

                print(
                    "Reason: Unsupported question "
                    "returned documents."
                )


    # =====================================================
    # FINAL METRICS
    # =====================================================

    overall_score = (
        passed_questions
        / total_questions
        * 100
        if total_questions > 0
        else 0
    )


    supported_accuracy = (
        supported_passed
        / supported_questions
        * 100
        if supported_questions > 0
        else 0
    )


    unsupported_rejection_rate = (
        unsupported_rejected
        / unsupported_questions
        * 100
        if unsupported_questions > 0
        else 0
    )


    # =====================================================
    # FINAL RESULT
    # =====================================================

    print("\n")

    print("=" * 70)

    print(
        "                    EVALUATION RESULT"
    )

    print("=" * 70)


    print(
        f"Total Questions              : "
        f"{total_questions}"
    )

    print(
        f"Passed                       : "
        f"{passed_questions}"
    )

    print(
        f"Failed                       : "
        f"{total_questions - passed_questions}"
    )

    print(
        f"Overall Score                : "
        f"{overall_score:.2f}%"
    )

    print(
        f"Supported Questions          : "
        f"{supported_questions}"
    )

    print(
        f"Supported Accuracy           : "
        f"{supported_accuracy:.2f}%"
    )

    print(
        f"Unsupported Questions        : "
        f"{unsupported_questions}"
    )

    print(
        f"Unsupported Rejection Rate   : "
        f"{unsupported_rejection_rate:.2f}%"
    )

    print("=" * 70)


# =========================================================
# LANGSMITH EVALUATION
# =========================================================


# ---------------------------------------------------------
# LangSmith Evaluator
# ---------------------------------------------------------

def langsmith_evaluator(
    run,
    example,
) -> dict:
    """
    LangSmith evaluator.

    `run` contains the actual chatbot output.

    `example` contains the expected evaluation output.
    """


    # -----------------------------------------------------
    # Get actual chatbot output
    # -----------------------------------------------------

    actual = run.outputs or {}


    actual_answer = actual.get(
        "answer",
        "",
    )


    actual_sources = actual.get(
        "sources",
        [],
    )


    # -----------------------------------------------------
    # Get expected evaluation output
    # -----------------------------------------------------

    expected = example.outputs or {}


    expected_keywords = expected.get(
        "expected_keywords",
        [],
    )


    expected_sources = expected.get(
        "relevant_sources",
        [],
    )


    question_type = expected.get(
        "type",
        "",
    )


    # =====================================================
    # SUPPORTED QUESTION
    # =====================================================

    if question_type == "supported":


        # -------------------------------------------------
        # Check answer
        # -------------------------------------------------

        answer_pass = check_answer(
            actual_answer,
            expected_keywords,
        )


        # -------------------------------------------------
        # Check sources
        # -------------------------------------------------

        source_pass = check_sources(
            actual_sources,
            expected_sources,
        )


        # -------------------------------------------------
        # Final score
        # -------------------------------------------------

        final_pass = (
            answer_pass
            and source_pass
        )


        return {
            "key": "rag_correctness",

            "score": (
                1
                if final_pass
                else 0
            ),

            "comment": (
                f"Answer: "
                f"{'PASS' if answer_pass else 'FAIL'}, "
                f"Sources: "
                f"{'PASS' if source_pass else 'FAIL'}"
            ),
        }


    # =====================================================
    # UNSUPPORTED QUESTION
    # =====================================================

    if question_type == "unsupported":

        rejected = (
            len(actual_sources) == 0
        )


        return {
            "key": "unsupported_rejection",

            "score": (
                1
                if rejected
                else 0
            ),

            "comment": (
                "Correctly rejected "
                "unsupported question."
                if rejected
                else
                "Returned sources for "
                "unsupported question."
            ),
        }


    # =====================================================
    # UNKNOWN QUESTION TYPE
    # =====================================================

    return {
        "key": "rag_correctness",
        "score": 0,
        "comment": "Unknown question type.",
    }


# ---------------------------------------------------------
# Create LangSmith Dataset
# ---------------------------------------------------------

def create_langsmith_dataset(
    client: Client,
):

    dataset_name = (
        "MemoryFAQ-RAG Evaluation"
    )


    # -----------------------------------------------------
    # Check whether dataset already exists
    # -----------------------------------------------------

    try:

        dataset = client.read_dataset(
            dataset_name=dataset_name
        )


        print(
            "\nLangSmith dataset already exists:"
        )

        print(
            dataset_name
        )


        return dataset


    except Exception:

        print(
            "\nCreating LangSmith dataset:"
        )

        print(
            dataset_name
        )


        # -------------------------------------------------
        # Create dataset
        # -------------------------------------------------

        dataset = client.create_dataset(
            dataset_name=dataset_name,

            description=(
                "Evaluation dataset for "
                "MemoryFAQ-RAG."
            ),
        )


        # -------------------------------------------------
        # Add evaluation examples
        # -------------------------------------------------

        for test_case in EVALUATION_DATASET:

            client.create_example(

                inputs={
                    "question":
                        test_case["question"],
                },

                outputs={
                    "expected_keywords":
                        test_case[
                            "expected_keywords"
                        ],

                    "relevant_sources":
                        test_case[
                            "relevant_sources"
                        ],

                    "type":
                        test_case["type"],
                },

                dataset_id=dataset.id,
            )


        print(
            "\nLangSmith dataset created successfully."
        )


        return dataset


# ---------------------------------------------------------
# LangSmith Evaluation Target
# ---------------------------------------------------------

def create_langsmith_target():

    print(
        "\nInitializing chatbot for "
        "LangSmith evaluation..."
    )


    chatbot = ChatbotService()


    def target(
        inputs: dict,
    ) -> dict:
        """
        Send one LangSmith dataset question
        to the MemoryFAQ-RAG chatbot.
        """

        question = inputs[
            "question"
        ]


        result = chatbot.ask(
            question
        )


        return {
            "answer":
                result["answer"],

            "sources":
                result["sources"],
        }


    return target


# ---------------------------------------------------------
# Run LangSmith Evaluation
# ---------------------------------------------------------

def run_langsmith_evaluation():

    print("\n")

    print("=" * 70)

    print(
        "              LANGSMITH RAG EVALUATION"
    )

    print("=" * 70)


    # -----------------------------------------------------
    # Create LangSmith client
    # -----------------------------------------------------

    client = Client()


    print(
        "\nConnected to LangSmith."
    )


    # -----------------------------------------------------
    # Create / load dataset
    # -----------------------------------------------------

    dataset = create_langsmith_dataset(
        client
    )


    # -----------------------------------------------------
    # Create evaluation target
    # -----------------------------------------------------

    target = create_langsmith_target()


    # -----------------------------------------------------
    # Run evaluation
    # -----------------------------------------------------

    print(
        "\nStarting LangSmith evaluation..."
    )


    results = client.evaluate(

        target,

        data=dataset,

        evaluators=[
            langsmith_evaluator,
        ],

        experiment_prefix=(
            "MemoryFAQ-RAG"
        ),
    )


    # -----------------------------------------------------
    # Finished
    # -----------------------------------------------------

    print(
        "\nLangSmith evaluation completed."
    )


    print(
        "\nOpen LangSmith and check the "
        "MemoryFAQ-RAG evaluation experiment."
    )


    return results


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # Run existing local evaluation
    # -----------------------------------------------------

    evaluate_rag()


    # -----------------------------------------------------
    # Run LangSmith evaluation
    # -----------------------------------------------------

    run_langsmith_evaluation()
