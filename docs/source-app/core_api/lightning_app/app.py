import lightning as L
from lightning.app.utilities.app_helpers import pretty_state


class Work(L.LightningWork):
    def __init__(self):
        super().__init__(cache_calls=False)
        # Attributes are registered automatically in the state.
        self.counter = 0

    def run(self):
        # Incrementing an attribute gets reflected in the `Flow` state.
        self.counter += 1


class Flow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.w = Work()

    def run(self):
        if self.w.has_started:
            print(f"State: {pretty_state(self.state)} \n")
        self.w.run()


app = L.LightningApp(Flow())
