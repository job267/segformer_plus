"""Small sequential queue: completed archives are required before advancing."""


def run_sequence(items, launch, already_complete, on_state):
    completed = []
    for item in items:
        if already_complete(item):
            completed.append(item)
            on_state("skipped_completed", item, completed.copy())
            continue
        on_state("running", item, completed.copy())
        code = launch(item)
        if code != 0:
            raise RuntimeError(f"{item} exited with code {code}; later experiments were not started")
        if not already_complete(item):
            raise RuntimeError(f"{item} exited without a verified completed archive")
        completed.append(item)
        on_state("completed_experiment", item, completed.copy())
    return completed
