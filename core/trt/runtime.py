class TrtRuntime:
    """Reserved TensorRT runtime adapter."""

    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        raise NotImplementedError("TODO(tensorrt): implement TensorRT runtime adapter.")

