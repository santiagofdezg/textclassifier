
class Error(Exception):
    """
    Base class for exceptions in this module.
    """
    pass


class NotFittedAlgorithmError(Error):
    """
    Exception raised when using a algorithm that was not fitted.
    """

    def __init__(self, message):
        # self.expression = expression
        self.message = message


class NotAvailableModelError(Error):
    """
    Exception raised when trying to load a non-existing model.
    """

    def __init__(self, message):
        # self.expression = expression
        self.message = message