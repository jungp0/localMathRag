from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re


DISPLAY_PATTERNS = [
    re.compile(r"\$\$(?P<body>.+?)\$\$", re.DOTALL),
    re.compile(r"\\\[(?P<body>.+?)\\\]", re.DOTALL),
    re.compile(r"\\begin\{equation\*?\}(?P<body>.+?)\\end\{equation\*?\}", re.DOTALL),
    re.compile(r"\\begin\{align\*?\}(?P<body>.+?)\\end\{align\*?\}", re.DOTALL),
]
INLINE_PATTERN = re.compile(r"(?<!\$)\$(?P<body>[^$\n]{3,240})\$(?!\$)")

MATH_SIGNAL_PATTERN = re.compile(
    r"(\\frac|\\sum|\\prod|\\int|\\sqrt|\\sigma|\\Sigma|\\mu|\\alpha|\\beta|"
    r"\\gamma|\\theta|\\lambda|\\operatorname|_\{|[\^_=]|[A-Z]_[A-Za-z0-9{])"
)
SECTION_EQUATION_NO_PATTERN = re.compile(r"(?:\bequation\b|\beq\.?|公式)\s*[\(:#]\s*([A-Za-z0-9_.-]+)", re.I)

LATEX_COMMANDS_NOT_SYMBOLS = {
    "begin",
    "end",
    "frac",
    "sqrt",
    "left",
    "right",
    "cdot",
    "times",
    "operatorname",
    "mathrm",
    "mathbf",
    "mathit",
    "mathbb",
    "text",
    "hat",
    "bar",
    "tilde",
    "dot",
    "ddot",
    "sum",
    "prod",
    "int",
    "exp",
    "log",
    "sin",
    "cos",
    "tan",
    "min",
    "max",
    "argmin",
    "argmax",
}

VARIABLE_WORDS = {
    "cov",
    "var",
    "diag",
    "trace",
    "transpose",
    "kalman",
}


@dataclass(slots=True)
class FormulaCandidate:
    latex: str
    normalized_latex: str
    symbols: list[str]
    operators: list[str]
    start: int
    end: int
    equation_no: str | None = None


def stable_id(prefix: str, *parts: str, length: int = 12) -> str:
    digest = hashlib.sha1("::".join(parts).encode("utf-8", errors="ignore")).hexdigest()
    return f"{prefix}.{digest[:length]}"


def strip_formula_delimiters(text: str) -> str:
    value = text.strip()
    wrappers = [
        ("$$", "$$"),
        ("\\[", "\\]"),
        ("\\(", "\\)"),
    ]
    for left, right in wrappers:
        if value.startswith(left) and value.endswith(right):
            return value[len(left) : -len(right)].strip()
    return value


def normalize_latex(latex: str) -> str:
    value = strip_formula_delimiters(latex)
    value = value.replace("\u2212", "-")
    value = value.replace("\u00d7", r"\times")
    value = value.replace("\u2211", r"\sum")
    value = value.replace("\u03bc", r"\mu")
    value = value.replace("\u03c3", r"\sigma")
    value = re.sub(r"\\left\s*", "", value)
    value = re.sub(r"\\right\s*", "", value)
    value = re.sub(r"\\,", "", value)
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", value)
    value = re.sub(r"\\mathbf\{([^{}]+)\}", r"\1", value)
    return value


def has_math_signal(text: str) -> bool:
    return bool(MATH_SIGNAL_PATTERN.search(text))


def extract_symbols(latex: str) -> list[str]:
    value = strip_formula_delimiters(latex)
    value = value.replace("\u2212", "-")
    value = re.sub(r"\\left\s*|\\right\s*|\\,", "", value)
    value = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", value)
    value = re.sub(r"\\mathbf\{([^{}]+)\}", r"\1", value)
    value = re.sub(r"(\\cdot|\\times|[=+*/(),\[\]])", " ", value)
    value = re.sub(r"(?<=[}A-Za-z0-9])(?=[A-Z][A-Za-z]?_)", " ", value)
    symbols: set[str] = set()

    for match in re.finditer(r"\\([A-Za-z]+)(?:_\{[^{}]+\}|_[A-Za-z0-9|+\-]+)?", value):
        command = match.group(1)
        if command not in LATEX_COMMANDS_NOT_SYMBOLS:
            symbols.add(match.group(0))

    variable_pattern = re.compile(
        r"(?<![A-Za-z\\])([A-Za-z][A-Za-z0-9]*"
        r"(?:_\{[^{}]+\}|_[A-Za-z0-9|+\-]+)?"
        r"(?:\^\{[^{}]+\}|\^[A-Za-z0-9+\-]+)?)(?![A-Za-z])"
    )
    for match in variable_pattern.finditer(value):
        token = match.group(1)
        bare = re.sub(r"[_^].*$", "", token)
        if bare.lower() in LATEX_COMMANDS_NOT_SYMBOLS or bare.lower() in VARIABLE_WORDS:
            continue
        if len(bare) > 3 and token == bare and not bare.isupper():
            continue
        symbols.add(token)

    return sorted(symbols, key=lambda item: (len(item), item))


def extract_operators(latex: str) -> list[str]:
    value = normalize_latex(latex)
    operators: set[str] = set()
    checks = {
        "add": ["+"],
        "subtract": ["-"],
        "multiply": [r"\cdot", r"\times", "*"],
        "divide": [r"\frac", "/"],
        "transpose": ["^T", "^{T}", r"^\top", r"^{\\top}"],
        "inverse": ["^{-1}", "^(-1)"],
        "sum": [r"\sum"],
        "sqrt": [r"\sqrt"],
        "covariance": ["cov", "Cov", "P_"],
        "variance": ["var", "Var", r"\sigma"],
    }
    for name, needles in checks.items():
        if any(needle in value for needle in needles):
            operators.add(name)
    if "=" in value:
        operators.add("equals")
    return sorted(operators)


def extract_equation_no(context: str) -> str | None:
    match = SECTION_EQUATION_NO_PATTERN.search(context)
    return match.group(1) if match else None


def extract_formula_candidates(text: str) -> list[FormulaCandidate]:
    candidates: list[FormulaCandidate] = []
    spans: list[tuple[int, int]] = []

    def add_candidate(body: str, start: int, end: int, context: str) -> None:
        latex = strip_formula_delimiters(body)
        if not latex or not has_math_signal(latex):
            return
        normalized = normalize_latex(latex)
        if len(normalized) < 3:
            return
        if any(abs(start - old_start) < 3 and abs(end - old_end) < 3 for old_start, old_end in spans):
            return
        spans.append((start, end))
        candidates.append(
            FormulaCandidate(
                latex=latex,
                normalized_latex=normalized,
                symbols=extract_symbols(latex),
                operators=extract_operators(latex),
                start=start,
                end=end,
                equation_no=extract_equation_no(context),
            )
        )

    for pattern in DISPLAY_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            context = text[max(0, start - 120) : min(len(text), end + 120)]
            add_candidate(match.group("body"), start, end, context)

    for match in INLINE_PATTERN.finditer(text):
        start, end = match.span()
        context = text[max(0, start - 120) : min(len(text), end + 120)]
        add_candidate(match.group("body"), start, end, context)

    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        start = offset + line.find(stripped) if stripped else offset
        end = start + len(stripped)
        offset += len(line)
        if not stripped or "=" not in stripped or len(stripped) > 320:
            continue
        if not has_math_signal(stripped):
            continue
        if any(start >= old_start and end <= old_end for old_start, old_end in spans):
            continue
        context = text[max(0, start - 120) : min(len(text), end + 120)]
        add_candidate(stripped, start, end, context)

    return candidates


def classify_domain(latex: str, context: str = "") -> list[str]:
    value = f"{normalize_latex(latex)} {context}".lower()
    domains: list[str] = []
    if "kalman" in value or "kalman" in context.lower():
        domains.append("kalman_filter")
    if "covariance" in value or "covariance" in context.lower() or "协方差" in context:
        domains.append("covariance")
    if "variance" in value or "方差" in context:
        domains.append("variance")
    if "noise" in value or "噪声" in context:
        domains.append("noise_model")

    compact = normalize_latex(latex)
    symbols = set(extract_symbols(compact))
    if {"P", "F", "Q"}.issubset({re.sub(r"[_^].*$", "", item) for item in symbols}) and "transpose" in extract_operators(compact):
        domains.extend(["kalman_filter", "covariance_prediction"])
    if "K" in symbols and "H" in "".join(symbols) and ("R" in symbols or "R_" in compact) and "inverse" in extract_operators(compact):
        domains.extend(["kalman_filter", "kalman_gain"])
    if "I" in symbols and "K" in symbols and "H" in "".join(symbols) and compact.startswith("P"):
        domains.extend(["kalman_filter", "covariance_update"])
    if "x" in {re.sub(r"[_^].*$", "", item) for item in symbols} and "F" in compact:
        domains.extend(["state_space_model"])

    deduped: list[str] = []
    for domain in domains:
        if domain not in deduped:
            deduped.append(domain)
    return deduped


def infer_variable_roles(latex: str, context: str = "") -> dict[str, dict[str, str]]:
    symbols = extract_symbols(latex)
    roles: dict[str, dict[str, str]] = {}
    normalized = normalize_latex(latex)
    for symbol in symbols:
        bare = re.sub(r"[_^].*$", "", symbol)
        role: str | None = None
        if bare == "P":
            if "k-1|k-1" in symbol:
                role = "previous_covariance"
            elif "|k-1" in symbol:
                role = "predicted_covariance"
            elif "|k}" in symbol or "|k)" in symbol or symbol.endswith("|k"):
                role = "updated_covariance"
            elif "pred" in context.lower() or "prediction" in context.lower():
                role = "predicted_covariance"
            elif "update" in context.lower() or "posterior" in context.lower():
                role = "updated_covariance"
            else:
                role = "state_covariance"
        elif bare == "F":
            role = "state_transition_matrix"
        elif bare == "Q":
            role = "process_noise_covariance"
        elif bare == "R":
            role = "measurement_noise_covariance"
        elif bare == "H":
            role = "measurement_matrix"
        elif bare == "K":
            role = "kalman_gain"
        elif bare == "x":
            role = "state_vector"
        elif bare == "z":
            role = "measurement_vector"
        elif bare == "u":
            role = "control_input"
        elif bare == "B":
            role = "control_matrix"
        elif bare in {"I", "1"}:
            role = "identity_matrix"
        elif bare in {"sigma", r"\sigma"} or r"\sigma" in symbol:
            role = "standard_deviation_or_variance_term"
        if role:
            roles[symbol] = {"role": role}

    if normalized.startswith("K") and "kalman_gain" not in [item.get("role") for item in roles.values()]:
        roles.setdefault("K", {"role": "kalman_gain"})
    return roles


def infer_assumptions(latex: str, context: str = "") -> list[str]:
    text = f"{latex} {context}".lower()
    assumptions: list[str] = []
    if "kalman" in text or {"F", "P", "Q"}.issubset({re.sub(r"[_^].*$", "", s) for s in extract_symbols(latex)}):
        assumptions.append("linear_discrete_system")
    if "gaussian" in text or "normal" in text or "高斯" in context:
        assumptions.append("gaussian_noise")
    if "independent" in text or "independence" in text or "独立" in context:
        assumptions.append("independent_noise")
    return assumptions
