# This file contains the fixed, simulated knowledge base for the RAG state.

KNOWLEDGE_BASE = [
    {
        "id": 1,
        "document": "ADA Guideline 2024: Diagnosis of Type 2 Diabetes (T2D) requires Fasting Plasma Glucose (FPG) >= 126 mg/dL, or A1C >= 6.5%. Symptoms include polyuria, polydipsia, and unexplained weight loss.",
        "tags": ["Diagnosis", "T2D", "FPG", "A1C"]
    },
    {
        "id": 2,
        "document": "Risk factors for Type 2 Diabetes include a BMI over 30, physical inactivity, first-degree relative with diabetes, and history of cardiovascular disease.",
        "tags": ["Risk", "T2D", "BMI", "CVD"]
    },
    {
        "id": 3,
        "document": "Initial treatment for T2D typically involves lifestyle modification (diet and exercise) followed by metformin, provided there are no contraindications.",
        "tags": ["Treatment", "Metformin"]
    },
    {
        "id": 4,
        "document": "The input of 'high blood sugar' strongly suggests the need for diagnostic screening according to Section 3.1 of the latest clinical guidance.",
        "tags": ["Screening", "Symptoms"]
    },
]