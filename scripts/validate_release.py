"""Read-only static checks of the publishable source tree."""
import ast
import json
import re
from release_utils import ROOT, assert_profile, load


PRIVATE_LOCATOR = re.compile(
    r"(?<![\w/])/(?:home|users|root|mnt|scratch|data)/[\w.~\-]+"
    r"|(?<![\w/])[a-z]:[/\\][\w.~\-]+"
    r"|(?<![\w.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![\w.])"
    r"|ssh://[\w.@\-]+",
    re.IGNORECASE,
)
CREDENTIAL = re.compile(
    r"(?:gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{25,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)


def _string_literal(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _string_literal(node.left)
        right = _string_literal(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def validate():
    problems = []
    count = 0
    excluded = {'.git', 'runs', '__pycache__', '.venv'}
    artifact_suffixes = {'.pt', '.pth', '.bin', '.safetensors', '.log', '.jsonl', '.pem', '.key'}
    text_suffixes = {'.py', '.json', '.md', '.txt', '.yml', '.yaml', '.cff'}
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if not path.is_file() or excluded.intersection(relative.parts):
            continue
        if path.suffix in artifact_suffixes:
            problems.append(f'Non-source artifact: {relative}')
        if path.suffix not in text_suffixes and path.name != 'NOTICE':
            continue
        source = path.read_text(encoding='utf-8')
        searchable_text = [source]
        if path.suffix == '.py':
            count += 1
            try:
                tree = ast.parse(source, filename=str(relative))
            except SyntaxError as error:
                problems.append(str(error))
            else:
                for node in ast.walk(tree):
                    literal = _string_literal(node)
                    if literal:
                        searchable_text.append(literal)
        if any(PRIVATE_LOCATOR.search(text) for text in searchable_text):
            problems.append(f'Possible private path or host: {relative}')
        if any(CREDENTIAL.search(text) for text in searchable_text):
            problems.append(f'Possible credential: {relative}')
    for path in (ROOT/'configs/methods').glob('*.json'):
        try:
            assert_profile(load(path)['training'])
        except ValueError as error:
            problems.append(f'{path.name}: {error}')
    return dict(
        python_files=count,
        methods=len(list((ROOT/'configs/methods').glob('*.json'))),
        problems=problems,
        passed=not problems,
    )


if __name__ == '__main__':
    result = validate()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
