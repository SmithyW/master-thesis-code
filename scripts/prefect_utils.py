from datetime import datetime


def generate_run_name(name: str):
    """Generate a task or flow run name based on the current date."""
    return f"{name}-run-on-{datetime.now():%A}"
