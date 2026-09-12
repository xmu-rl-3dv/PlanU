"""Validate an interpreter's installed distributions against an exact lock."""

import argparse
import json
from importlib.metadata import distributions
from pathlib import Path
import re
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname


def normalize_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def load_exact_lock(path):
    expected = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise ValueError(
                "distribution lock contains a non-exact requirement: " + line
            )
        raw_name, version = line.split("==", 1)
        name = normalize_name(raw_name.strip())
        version = version.strip()
        if not name or not version:
            raise ValueError(
                "distribution lock contains an invalid requirement: " + line
            )
        if name in expected:
            raise ValueError(
                "distribution lock contains duplicate normalized name: " + name
            )
        expected[name] = version
    if not expected:
        raise ValueError("distribution lock is empty")
    return expected


def installed_distribution_records():
    records = {}
    for distribution in distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        name = normalize_name(raw_name)
        prior = records.get(name)
        if prior is not None:
            raise ValueError(
                "duplicate installed distributions for {}: {} and {}".format(
                    name,
                    prior.version,
                    distribution.version,
                )
            )
        records[name] = distribution
    return records


def _editable_origin(distribution):
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        try:
            direct = json.loads(direct_url)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "invalid direct_url.json for editable distribution"
            ) from exc
        if direct.get("dir_info", {}).get("editable") is not True:
            return None
        parsed = urlparse(direct.get("url", ""))
        if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
            return None
        return Path(url2pathname(unquote(parsed.path))).resolve()

    metadata_path = Path(getattr(distribution, "_path", ""))
    if metadata_path.name.endswith(".egg-info"):
        return Path(distribution.locate_file("")).resolve()
    return None


def verify_distribution_lock(
    lock_path,
    *,
    allowed_extras=(),
    required_editables=None,
):
    expected = load_exact_lock(lock_path)
    records = installed_distribution_records()
    installed = {
        name: str(distribution.version)
        for name, distribution in records.items()
    }
    allowed = {normalize_name(name) for name in allowed_extras}
    editables = {
        normalize_name(name): Path(origin).resolve()
        for name, origin in (required_editables or {}).items()
    }
    exemptions = allowed | set(editables)
    overlap = set(expected) & exemptions
    if overlap:
        raise ValueError(
            "lock entries may not also be exempt: {}".format(
                ", ".join(sorted(overlap))
            )
        )

    missing = sorted(set(expected) - set(installed))
    mismatches = sorted(
        (
            name,
            expected[name],
            installed[name],
        )
        for name in set(expected) & set(installed)
        if expected[name] != installed[name]
    )
    extras = sorted(set(installed) - set(expected) - exemptions)
    problems = []
    if missing:
        problems.append("missing={}".format(missing))
    if mismatches:
        problems.append(
            "version mismatches={}".format(
                [
                    "{} expected {} found {}".format(name, wanted, actual)
                    for name, wanted, actual in mismatches
                ]
            )
        )
    if extras:
        problems.append("extras={}".format(extras))

    for name, expected_origin in sorted(editables.items()):
        distribution = records.get(name)
        if distribution is None:
            problems.append("missing editable={}".format(name))
            continue
        actual_origin = _editable_origin(distribution)
        if actual_origin != expected_origin:
            problems.append(
                "editable {} expected origin {} found {}".format(
                    name,
                    expected_origin,
                    actual_origin or "non-editable",
                )
            )

    if problems:
        raise ValueError("distribution lock mismatch: " + "; ".join(problems))
    return dict(sorted(installed.items()))


def _editable_argument(value):
    name, separator, origin = value.partition("=")
    if not separator or not name or not origin:
        raise argparse.ArgumentTypeError(
            "editable exemption must use NAME=PATH"
        )
    return name, origin


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("lock_path")
    parser.add_argument("--allow-extra", action="append", default=[])
    parser.add_argument(
        "--editable-extra",
        action="append",
        default=[],
        type=_editable_argument,
    )
    args = parser.parse_args(argv)
    try:
        verify_distribution_lock(
            args.lock_path,
            allowed_extras=args.allow_extra,
            required_editables=dict(args.editable_extra),
        )
    except ValueError as exc:
        parser.exit(1, "{}\n".format(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
