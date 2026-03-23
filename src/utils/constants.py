# Tokens that suggest engineering/tool-heavy job requirements (substring match on normalized tokens).
TECH_SIGNAL_TOKENS = frozenset(
    {
        "python",
        "java",
        "javascript",
        "typescript",
        "golang",
        "rust",
        "ruby",
        "php",
        "swift",
        "kotlin",
        "scala",
        "sql",
        "aws",
        "gcp",
        "azure",
        "kubernetes",
        "k8s",
        "docker",
        "terraform",
        "ansible",
        "jenkins",
        "react",
        "angular",
        "vue",
        "node",
        "kafka",
        "redis",
        "mongodb",
        "postgres",
        "mysql",
        "graphql",
        "rest",
        "api",
        "saas",
        "microservice",
        "devops",
        "ci",
        "cd",
        "machine",
        "learning",
        "llm",
        "langchain",
        "tensorflow",
        "pytorch",
        "spark",
        "hadoop",
        "snowflake",
        "databricks",
        "software",
        "engineering",
        "backend",
        "frontend",
        "fullstack",
    }
)

DOC_PROMPT_MARKER_PREFIXES = (
    "---begin",
    "---end",
    "begin resume",
    "end resume",
    "begin jd",
    "end jd",
)

DOC_INJECTION_LINE_PATTERNS = (
    r"ignore\s+previous\s+instructions?",
    r"disregard\s+all\s+previous",
    r"follow\s+these\s+instructions",
    r"\byou\s+are\s+chatgpt\b",
    r"\bsystem\s*:",
    r"\bdeveloper\s*:",
    r"\bassistant\s*:",
    r"reveal\s+(the\s+)?(system|prompt|instructions?)",
    r"jailbreak",
    r"do\s+not\s+follow\s+your\s+rules",
)


# BIG NOTE: A lot of these constants are somewhat arbitrary based on web searches and AI brainstorming chats. These
# can and should be tweaked based on actual production data and feedback.
class Constants:
    # Primary Jaccard threshold for normalized skill phrase vs resume skill phrase.
    SKILL_MATCH_JACCARD_THRESHOLD = 0.42

    # When at least this fraction of JD requirement tokens appear in the resume token bag, count as match.
    SKILL_TOKEN_COVERAGE_THRESHOLD = 0.5
    SKILL_TOKEN_COVERAGE_THRESHOLD_LONG = 0.34

    # Neutral experience match when JD does not state years (avoid inflating final score).
    EXPERIENCE_MATCH_NEUTRAL = 0.6

    # Chunking for extraction LLM calls (characters).
    CHUNK_MAX_CHARS = 6000
    CHUNK_OVERLAP_CHARS = 400

    DOC_SANITIZE_MAX_LINE_LEN = 1200
    DOC_SANITIZE_MAX_REPEAT_CHAR_SEQ = 10
    DOC_SANITIZE_MAX_CONSECUTIVE_BLANK_LINES = 2

    # Dynamic final score: when JD competency list looks tech-heavy, skills weigh more; otherwise role+exp weigh more.
    WEIGHTS_TECH_HEAVY = (0.45, 0.28, 0.27)  # skills, exp, role (no JD education reqs)
    WEIGHTS_BALANCED = (0.36, 0.34, 0.30)

    # When JD lists education requirements, include education_match_score with these 4-tuples (skills, exp, role, edu).
    WEIGHTS_TECH_HEAVY_4 = (0.40, 0.22, 0.22, 0.16)
    WEIGHTS_BALANCED_4 = (0.30, 0.27, 0.23, 0.20)

    # Fraction of JD requirements that must hit tech-signal tokens to use WEIGHTS_TECH_HEAVY.
    JD_TECH_RATIO_THRESHOLD = 0.25

    # Requirement-constraint and judge integration.
    CONSTRAINT_SCORE_BLEND_WEIGHT = 0.20
    HARD_REQUIREMENT_MISS_PENALTY = 0.15
    HARD_REQUIREMENT_MISS_PENALTY_CAP = 0.35
