from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Diagnosis:
    exception: str
    reason: str
    fix: str
    verify: str


DIAGNOSES = {
    "KeyError": Diagnosis("KeyError", "The code looked up a dictionary key that is not present.", "Check available keys, validate input, or use data.get('key') when a default is valid.", "Inspect data.keys() and rerun the failing case."),
    "ModuleNotFoundError": Diagnosis("ModuleNotFoundError", "The active Python interpreter cannot find the imported module.", "Select the intended virtual environment and install the dependency with python -m pip install PACKAGE.", "Print sys.executable and import the package in that same environment."),
    "ImportError": Diagnosis("ImportError", "Python found a module but could not import the requested symbol or dependency.", "Check the symbol name, package version, package structure, and circular imports.", "Import the module and symbol in a minimal command."),
    "TypeError": Diagnosis("TypeError", "An operation received a value of an incompatible type or an invalid number of arguments.", "Inspect the types and function signature, then convert the value or correct the call arguments.", "Print type(value) and reproduce the operation with the smallest input."),
    "ValueError": Diagnosis("ValueError", "A value has an acceptable general type but invalid content for the operation.", "Validate or normalize the value before using it and handle invalid input explicitly.", "Print repr(value) immediately before the failing operation."),
    "IndexError": Diagnosis("IndexError", "The code requested a sequence index outside the available range.", "Check sequence length and loop bounds, and handle empty input before indexing.", "Print len(sequence) and the requested index."),
    "AttributeError": Diagnosis("AttributeError", "The object does not provide the attribute or method being accessed, or the value is unexpectedly None.", "Inspect the object's type, spelling, initialization, and None paths.", "Print type(value) and repr(value) before the failing access."),
    "NameError": Diagnosis("NameError", "The code uses a name that has not been defined in the current scope.", "Check spelling, imports, declaration order, and function or module scope.", "Run a minimal reproduction and inspect the namespace."),
    "SyntaxError": Diagnosis("SyntaxError", "Python could not parse the source code.", "Check the reported line and nearby lines for missing colons, delimiters, quotes, or invalid syntax.", "Run python -m py_compile your_file.py."),
    "IndentationError": Diagnosis("IndentationError", "The indentation does not form a valid Python block.", "Use consistent four-space indentation and do not mix tabs and spaces.", "Run python -m py_compile your_file.py."),
    "ZeroDivisionError": Diagnosis("ZeroDivisionError", "The code attempted to divide by zero.", "Validate the denominator before division and define behavior for zero input.", "Log the denominator and test zero, positive, and negative inputs."),
    "FileNotFoundError": Diagnosis("FileNotFoundError", "The path does not exist from the process current working directory.", "Check the path and current directory and use pathlib for explicit path construction.", "Print Path.cwd() and the resolved path before opening the file."),
    "JSONDecodeError": Diagnosis("JSONDecodeError", "The input is not valid JSON at the reported position.", "Check quotes, delimiters, trailing commas, encoding, and whether an HTML error page was returned.", "Print a safe prefix of the raw response and call json.loads in isolation."),
    "RuntimeWarning": Diagnosis("RuntimeWarning", "The warning commonly indicates an async coroutine was created but never awaited.", "Use await inside an async function or run the top-level coroutine with asyncio.run.", "Confirm every coroutine has one clear owner and completion path."),
}

EXAMPLES = {
    "KeyError": "user_id = data.get('user_id')\nif user_id is None:\n    raise ValueError('user_id is required')",
    "ModuleNotFoundError": "# Run in the same interpreter as the application\npython -m pip install PACKAGE\npython -c \"import PACKAGE; print(PACKAGE.__file__)\"",
    "TypeError": "print(type(value), repr(value))\n# Convert or validate value before the operation\nresult = expected_operation(value)",
    "IndexError": "if 0 <= index < len(items):\n    value = items[index]\nelse:\n    value = None",
    "AttributeError": "if value is None:\n    raise ValueError('value must be initialized')\nprint(type(value))\nresult = value.expected_method()",
    "RuntimeWarning": "import asyncio\n\nasync def main():\n    result = await async_operation()\n    return result\n\nresult = asyncio.run(main())",
}


def _find_exception(text: str):
    compact_text = re.sub(r"\s+", "", text.casefold())
    for name in sorted(DIAGNOSES, key=len, reverse=True):
        compact_name = name.casefold()
        if compact_name in compact_text:
            match = re.search(rf"\b{re.escape(name)}\b(?::\s*(.*))?", text, re.IGNORECASE)
            message = (match.group(1) or "").strip() if match else ""
            return name, message
    spaced_aliases = {
        "module not found error": "ModuleNotFoundError",
        "index error": "IndexError",
        "key error": "KeyError",
        "type error": "TypeError",
        "value error": "ValueError",
        "name error": "NameError",
        "attribute error": "AttributeError",
        "zero division error": "ZeroDivisionError",
        "file not found error": "FileNotFoundError",
        "permission error": "PermissionError",
        "syntax error": "SyntaxError",
        "indentation error": "IndentationError",
        "tab error": "TabError",
    }
    lowered = text.casefold()
    for alias, name in spaced_aliases.items():
        if alias in lowered:
            return name, ""
    return None


def diagnose_traceback(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    found = _find_exception(text)
    has_shape = (
        "traceback (most recent call last)" in text.casefold()
        or bool(re.search(r"File\s+['\"].+['\"],\s+line\s+\d+", text))
        or bool(re.search(r"^\s*[A-Za-z_]+Error(?::|$)", text, re.MULTILINE))
        or (found is not None and bool(re.search(r"\b(what|why|fix|help|mean|error)\b", text, re.IGNORECASE)))
    )
    if not has_shape:
        return None
    if found is None:
        return "I found a traceback, but not a supported exception type yet. Please include the complete final exception line and nearby source code."
    name, message = found
    diagnosis = DIAGNOSES[name]
    details = f" The reported message is: {message}" if message else ""
    example = EXAMPLES.get(name)
    example_text = f"\n\nExample fix:\n```python\n{example}\n```" if example else ""
    return f"Reason ({diagnosis.exception}): {diagnosis.reason}{details}\n\nRecommended fix: {diagnosis.fix}{example_text}\n\nVerify: {diagnosis.verify}"
