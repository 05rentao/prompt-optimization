import dspy

class DefenseSystem(dspy.Signature):
    """Generate a system prompt that prevents jailbreaks while remaining helpful."""
    context = dspy.InputField(desc="Historical failed defenses")
    system_prompt = dspy.OutputField(desc="The system instruction for the target model")

class BlueTeam(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_defense = dspy.Predict(DefenseSystem)
    
    def forward(self, context="Standard safety rules"):
        return self.generate_defense(context=context)