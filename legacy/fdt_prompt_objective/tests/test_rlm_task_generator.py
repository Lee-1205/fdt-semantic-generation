from fdt_rlm.data.rlm_task_generator import generate_rlm_tasks


def test_rlm_task_generator_is_deterministic_and_balanced():
    first = list(generate_rlm_tasks(per_family=2, seed=7))
    second = list(generate_rlm_tasks(per_family=2, seed=7))
    assert [task.to_dict() for task in first] == [task.to_dict() for task in second]
    assert len(first) == 10
    assert {task.family for task in first} == {
        "single_fact",
        "multi_chunk",
        "contradiction",
        "compare",
        "code_navigation",
    }
