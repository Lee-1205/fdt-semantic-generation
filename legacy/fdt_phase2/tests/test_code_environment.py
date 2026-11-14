from fdt_rlm.runtime.actions import Action
from fdt_rlm.runtime.code_environment import CodeEnvironment
from fdt_rlm.runtime.recursive_controller import ModelStep, RecursiveController


def test_code_environment_is_read_only_and_finds_symbols():
    env = CodeEnvironment({"src/a.py": "from .b import VALUE\n\ndef target():\n    return VALUE\n", "src/b.py": "VALUE = 7\n"})
    found = env.execute(Action.from_mapping({"action": "FIND_SYMBOL", "name": "target"}))
    assert found.ok
    assert "src/a.py" in found.observation
    imports = env.execute(Action.from_mapping({"action": "FIND_IMPORTS", "path": "src/a.py"}))
    assert imports.ok
    assert "from .b import VALUE" in imports.observation


def test_controller_advertises_code_actions_and_uses_text_subcall():
    class Backend:
        def __init__(self):
            self.prompts = []
            self.outputs = iter(
                [
                    '{"action":"READ_FILE","path":"src/app.py","start":0,"end":2}',
                    '{"action":"CALL","question":"What is imported?","context_ref":"result_0"}',
                    '{"action":"STOP","answer":"os"}',
                    '{"action":"STOP","answer":"done"}',
                ]
            )

        def generate_action(self, prompt, anchor_state):
            self.prompts.append(prompt)
            return ModelStep(next(self.outputs))

    backend = Backend()
    controller = RecursiveController(backend)
    result = controller.run("Inspect it", CodeEnvironment({"src/app.py": "import os\nprint(os.name)"}))
    assert result.answer == "done"
    assert "READ_FILE" in backend.prompts[0]
    assert "READ, SEARCH" in backend.prompts[2]
