"""
Dummy module for loading legacy checkpoints from previous project.
This allows unpickling checkpoints that reference src.config_loader classes.

This module dynamically creates dummy classes for any attribute accessed,
allowing it to handle any config class from the old project.
"""

import sys


class _DummyConfigBase:
    """Base class for dummy config classes."""
    
    def __init__(self, *args, **kwargs):
        """Accept any arguments to handle various initialization patterns."""
        # Store all attributes dynamically
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __setstate__(self, state):
        """Handle unpickling - restore object state."""
        self.__dict__.update(state)
    
    def __getstate__(self):
        """Handle pickling - return object state."""
        return self.__dict__


# Pre-define common classes
class Config(_DummyConfigBase):
    """Dummy Config class for legacy checkpoint compatibility."""
    pass


class SolverConfig(_DummyConfigBase):
    """Dummy SolverConfig class for legacy checkpoint compatibility."""
    pass


class DataConfig(_DummyConfigBase):
    """Dummy DataConfig class for legacy checkpoint compatibility."""
    pass


class NCCConfig(_DummyConfigBase):
    """Dummy NCCConfig class for legacy checkpoint compatibility."""
    pass


class DatasetConfig(_DummyConfigBase):
    """Dummy DatasetConfig class for legacy checkpoint compatibility."""
    pass


def __getattr__(name):
    """
    Dynamically create any missing config class.
    This allows loading checkpoints with any config class name.
    """
    # Create a new class dynamically
    new_class = type(name, (_DummyConfigBase,), {
        '__module__': __name__,
        '__doc__': f'Dynamically created dummy {name} class for legacy checkpoint compatibility.'
    })
    
    # Add it to the module so it can be found next time
    setattr(sys.modules[__name__], name, new_class)
    
    return new_class

