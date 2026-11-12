"""Numerical check of the actual token-KL expression without GPU worker imports."""

import ast
from pathlib import Path

import numpy as np


class Tensor(np.ndarray):
    def float(self):
        return self


class Torch:
    @staticmethod
    def sum(value, dim):
        return value.sum(axis=dim)


def test_teacher_topk_mass_is_not_renormalized():
    path = Path(__file__).resolve().parents[1] / "verl/workers/actor/dp_actor.py"
    tree = ast.parse(path.read_text())
    assignment = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "token_kl" for target in node.targets)
    )
    teacher = np.array([[[0.6, 0.3]]]).view(Tensor)
    student = np.array([[[0.4, 0.2]]]).view(Tensor)
    expression = compile(ast.Expression(assignment.value), str(path), "eval")
    actual = eval(expression, {"__builtins__": {}}, {
        "torch": Torch, "teacher_probs": teacher,
        "teacher_topk_logps": np.log(teacher), "student_topk_logps": np.log(student),
    })
    # Normalizing both selected supports would make this example's KL zero.
    np.testing.assert_allclose(actual, [[0.9 * np.log(1.5)]], rtol=1e-7)
