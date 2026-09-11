class ModelNotFoundError(Exception):
  def __init__(self, *args: object) -> None:
    super().__init__(*args)
    
class ModelNotSpecifiedError(Exception):
  def __init__(self, *args: object) -> None:
    super().__init__(*args)