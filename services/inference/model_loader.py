import threading

import mlflow


class ModelHolder:
    def __init__(self):
        self.model = None
        self.prev_uri = None
        self.cur_uri = None
        self._lock = threading.Lock()

    def load(self, model_uri: str):
        with self._lock:
            self.prev_uri = self.cur_uri
            self.model = mlflow.pyfunc.load_model(model_uri)
            self.cur_uri = model_uri

    def rollback(self):
        if self.prev_uri:
            self.load(self.prev_uri)
            return True
        return False


holder = ModelHolder()
