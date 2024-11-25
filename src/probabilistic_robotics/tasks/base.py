


class BaseTask:

    def start(self, global_state):
        global_state["task_status"] = "running"
    
    def run_cycle(self, global_state):
        raise NotImplementedError
    
    def finish(self, global_state):
        global_state["current_task_id"] += 1
        global_state["task_status"] = "not_started"
