from fastapi import HTTPException

class CoordinateNotAllowException(HTTPException):
    def __init__(self):
        super().__init__(status_code=400, detail="Coordinates are out of allowed range")

class PathNotFoundException(HTTPException):
    def __init__(self):
        super().__init__(status_code=404, detail="No valid path found between the specified nodes")