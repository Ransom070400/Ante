"""Errors raised by the 0G backend."""


class ZeroGRequestError(RuntimeError):
    """A 0G provider rejected a request.

    Providers report refusals such as an unfunded sub-account in the body of an HTTP 200,
    so this is raised from the body rather than from the status code. Without it a
    refusal reads as an empty transcript and the robot goes silent for no visible reason.
    """
