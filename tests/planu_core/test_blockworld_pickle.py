import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BLOCKWORLD_ROOT = REPOSITORY_ROOT / "blockworld"


def _pythonpath():
    paths = [os.fspath(REPOSITORY_ROOT), os.fspath(BLOCKWORLD_ROOT)]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    return os.pathsep.join(paths)


def test_blockworld_state_pickle_loads_in_a_fresh_process(tmp_path):
    pickle_path = tmp_path / "state.pkl"
    environment = dict(os.environ, PYTHONPATH=_pythonpath())
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import pickle\n"
                "from pathlib import Path\n"
                "from planu_core import ActionCandidate, ActionNode, LanguageNode\n"
                "from planu_core.adapters.blockworld import BWStateRAP\n"
                "from planu_core.distribution import QuantileDistribution\n"
                "from reasoners.algorithm.planU import PlanUResult\n"
                "initial = BWStateRAP(0, '', 'before', '')\n"
                "final = BWStateRAP(1, 'before', 'after', 'pick up red')\n"
                "root = LanguageNode(initial, ('state', 'before'))\n"
                "candidate = ActionCandidate(\n"
                "    'pick up red', 'pick up red', 'pick up red',\n"
                "    {{'prior_score': 0.5}},\n"
                ")\n"
                "action = ActionNode(\n"
                "    root,\n"
                "    candidate,\n"
                "    QuantileDistribution.from_scalar(0.5, 3, -1, 1),\n"
                ")\n"
                "root.children[candidate.key] = action\n"
                "child = action.get_or_create_outcome(\n"
                "    final, ('state', 'after'), True, False,\n"
                ")\n"
                "result = PlanUResult(\n"
                "    final, 1.0, ([initial, final], ['pick up red']),\n"
                "    [root, child], root,\n"
                ")\n"
                "Path({!r}).write_bytes(pickle.dumps(result))\n"
            ).format(os.fspath(pickle_path)),
        ],
        check=True,
        env=environment,
        timeout=30,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import pickle\n"
                "from pathlib import Path\n"
                "result = pickle.loads(Path({!r}).read_bytes())\n"
                "assert result.terminal_state.blocks_state == 'after'\n"
                "assert result.trace[1] == ['pick up red']\n"
                "candidate = result.tree_state.children['pick up red'].action\n"
                "assert candidate.metadata == {{'prior_score': 0.5}}\n"
                "try:\n"
                "    candidate.metadata['prior_score'] = 1.0\n"
                "except TypeError:\n"
                "    pass\n"
                "else:\n"
                "    raise AssertionError('metadata became mutable')\n"
            ).format(os.fspath(pickle_path)),
        ],
        capture_output=True,
        env=environment,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
