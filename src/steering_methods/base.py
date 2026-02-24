from abc import ABC, abstractmethod
import torch

class SteeringMethod(ABC):  # Abstract base class for steering methods
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer 
        self.handle = None  # Storage for the PyTorch hook (to remove the hook later)

    @abstractmethod
    def apply(self, **kwargs):
        """
        Every child class MUST implement this.
        """
        pass

    def remove(self):
        """
        Standardized way to 'turn off' the steering.
        """
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
            print("Hook removed successfully.")
        else:
            print("No active hook to remove.")