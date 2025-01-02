class Response:
    def success(message: str = 'success', data: any = None):
        return {
            "status": 200,
            "message": message,
            "data": data
        }
    
    def bad_request(message: str = 'bad request', error: list = []):
        return {
            "status": 400,
            "message": message,
            "error": error
        }
    
    def internal_error(message: str = 'internal request', error: list = []):
        return {
            "status": 500,
            "message": message,
            "error": error
        }