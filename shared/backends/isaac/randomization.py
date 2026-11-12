"""Domain randomization is not bound on Isaac Lab 2; declarations are refused, not ignored."""
from shared.assembly.randomization import unsupported


def validate(randomizations, settings):
    unsupported('isaac', randomizations)
