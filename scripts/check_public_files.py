"""Audit the Git index before publishing; never print matched private values."""

import argparse
import ipaddress
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

TOP_LEVEL = {".gitignore", ".gitattributes", "README.md", "THIRD_PARTY_NOTICES.md",
             "requirements.txt", "pyproject.toml"}
LICENSES = {"SegFormer_LICENSE.txt", "CamVid_repository_LICENSE.txt", "sources.json"}
PATTERNS = {
    "local filesystem path": re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|/(?:Users|home|mnt)/)"),
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "access credential": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{35,}|sk-[A-Za-z0-9_-]{24,}|AKIA[0-9A-Z]{16})"),
    "credential in URL": re.compile(r"https?://[^\s/:]+:[^\s/@]+@"),
    "assigned credential": re.compile(r'''(?i)\b(?:api_key|access_token|password|passwd|secret_key)\b\s*[:=]\s*["'][^"'\n]{8,}["']'''),
}
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def allowed_file(name):
    parts = PurePosixPath(name).parts
    if name in TOP_LEVEL:
        return True
    if len(parts) == 2 and parts[0] in {"src", "scripts"}:
        return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.py", parts[1]))
    if len(parts) == 2 and parts[0] == "configs":
        return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.yaml", parts[1]))
    if len(parts) == 2 and parts[0] == "demo":
        return parts[1] in {"app.py", "index.html"}
    return len(parts) == 3 and parts[:2] == ("docs", "licenses") and parts[2] in LICENSES


def index_entries(root):
    output = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=root)
    for record in output.split(b"\0"):
        if record:
            metadata, name = record.split(b"\t", 1)
            mode, oid, stage = metadata.decode("ascii").split()
            yield name.decode("utf-8"), mode, oid, stage


def audit(root):
    entries = list(index_entries(root))
    issues = []
    for name, mode, oid, stage in entries:
        if stage != "0" or mode not in {"100644", "100755"}:
            issues.append({"file": name, "reason": "unsupported entry or merge conflict"})
            continue
        if not allowed_file(name):
            issues.append({"file": name, "reason": "outside the public source allowlist"})
            continue
        blob = subprocess.check_output(["git", "cat-file", "blob", oid], cwd=root)
        try:
            text = blob.decode("utf-8-sig")
        except UnicodeError:
            issues.append({"file": name, "reason": "non-text content"})
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                issues.append({"file": name, "reason": label})
        # Preserve third-party attribution exactly; do not remove license contacts.
        if not name.startswith("docs/licenses/") and EMAIL.search(text):
            issues.append({"file": name, "reason": "email outside third-party license"})
        for match in IPV4.finditer(text):
            try:
                address = ipaddress.IPv4Address(match.group())
            except ipaddress.AddressValueError:
                continue
            if address.is_private and not address.is_loopback and not address.is_unspecified:
                issues.append({"file": name, "reason": "private network address"})
                break
    return {"checked_files": len(entries), "passed": bool(entries) and not issues, "issues": issues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = audit(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
