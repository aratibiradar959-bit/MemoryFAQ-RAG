EVALUATION_DATASET = [

    # ==========================================
    # SUPPORTED QUESTIONS
    # ==========================================

    {
        "question": "How many days can employees work remotely?",
        "relevant_sources": [
            "remote_work_policy.txt",
            "company_info.txt",
        ],
        "expected_keywords": [
            "three days",
            "per week",
        ],
        "type": "supported",
    },

    {
        "question": "What are the normal working hours?",
        "relevant_sources": [
            "company_info.txt",
            "remote_work_policy.txt",
        ],
        "expected_keywords": [
            "9 AM",
            "6 PM",
        ],
        "type": "supported",
    },

    {
        "question": "How many paid leave days do employees receive?",
        "relevant_sources": [
            "leave_policy.txt",
            "company_info.txt",
        ],
        "expected_keywords": [
            "20",
            "paid leave",
        ],
        "type": "supported",
    },

    {
        "question": "Can remote employees take leave?",
        "relevant_sources": [
            "remote_work_policy.txt",
            "leave_policy.txt",
            "company_info.txt",
        ],
        "expected_keywords": [
            "leave",
            "remote",
        ],
        "type": "supported",
    },


    # ==========================================
    # UNSUPPORTED QUESTIONS
    # ==========================================

    {
        "question": "What is Apple's stock price?",
        "relevant_sources": [],
        "expected_keywords": [],
        "type": "unsupported",
    },

    {
        "question": "What is the weather today?",
        "relevant_sources": [],
        "expected_keywords": [],
        "type": "unsupported",
    },

    {
        "question": "Who is the CEO of Google?",
        "relevant_sources": [],
        "expected_keywords": [],
        "type": "unsupported",
    },
]