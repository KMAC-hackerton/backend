from fastapi import HTTPException

class InvalidRequestException(HTTPException):
    def __init__(self):
        super().__init__(status_code=400, detail="Invalid request parameters")


class PathNotFoundException(HTTPException):
    def __init__(self):
        super().__init__(status_code=404, detail="No valid path found between the specified nodes")