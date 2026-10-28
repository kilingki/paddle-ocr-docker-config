class APIError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message
        self.details = {}

    def body(self, **details):
        return {"error": {"code": self.code, "message": self.message, **self.details, **details}}
