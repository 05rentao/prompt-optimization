import torch
from tqdm import tqdm

from base import SteeringMethod
import torch

class ActAddSteering(SteeringMethod):
    def apply(self, layer_idx: int, steering_vector: torch.Tensor, coefficient: float = 1.0):
        """
        Injects the steering vector into the specified layer's forward pass.
        """
        # 1. Safety check: Remove any lingering hooks from previous runs
        self.remove()

        # 2. Prepare the vector (ensure it matches model device and dtype)
        vector = steering_vector.to(device=self.model.device, dtype=self.model.dtype)
        
        # 3. Define the intervention hook
        def steering_hook(module, input, output):
            # output is a tuple containing (hidden_states, past_key_values, etc.)
            hidden_states = output[0] 
            
            # The addition broadcasts across the seq_len dimension automatically:
            # hidden_states: [batch, seq_len, hidden_dim]
            # vector: [hidden_dim]
            modified_hidden_states = hidden_states + (coefficient * vector)
            
            # Return the tuple exactly how the next layer expects it
            return (modified_hidden_states,) + output[1:]

        # 4. Attach the hook to the target Qwen layer
        target_layer = self.model.model.layers[layer_idx]
        self.handle = target_layer.register_forward_hook(steering_hook)
        print(f"[*] Steering applied to Layer {layer_idx} | Coeff: {coefficient}")


def format_qwen_prompt(tokenizer, instruction, response):
    """Formats prompts using Qwen's specific ChatML template."""
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response}
    ]
    # tokenize=False returns the raw string with <|im_start|> and <|im_end|> tokens
    return tokenizer.apply_chat_template(messages, tokenize=False)

def extract_steering_vectors(model, tokenizer, pos_data, neg_data, target_layers):
    """
    pos_data / neg_data: List of dicts, e.g., [{"instruction": "...", "response": "..."}]
    target_layers: List of integers representing layer indices to extract.
    """
    vectors = {}
    
    def get_capture_hook(layer_idx, storage_dict, key):
        def hook(module, input, output):
            # output[0] is hidden_states: [batch, seq_len, hidden_dim]
            # We capture the activation of the LAST token
            last_token_act = output[0][:, -1, :].detach().cpu()
            storage_dict[key][layer_idx].append(last_token_act)
            return None # Do not modify the forward pass
        return hook

    # Storage for raw activations
    activations = {"pos": {l: [] for l in target_layers}, "neg": {l: [] for l in target_layers}}
    
    # Process Positive Data
    print("Extracting positive activations...")
    pos_handles = []
    for l in target_layers:
        handle = model.model.layers[l].register_forward_hook(get_capture_hook(l, activations, "pos"))
        pos_handles.append(handle)
        
    for item in tqdm(pos_data):
        prompt = format_qwen_prompt(tokenizer, item["instruction"], item["response"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
            
    for h in pos_handles: h.remove() # Clean up positive hooks

    # Process Negative Data
    print("Extracting negative activations...")
    neg_handles = []
    for l in target_layers:
        handle = model.model.layers[l].register_forward_hook(get_capture_hook(l, activations, "neg"))
        neg_handles.append(handle)
        
    for item in tqdm(neg_data):
        prompt = format_qwen_prompt(tokenizer, item["instruction"], item["response"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
            
    for h in neg_handles: h.remove() # Clean up negative hooks

    # Compute Means and Final Vectors
    for l in target_layers:
        pos_tensor = torch.cat(activations["pos"][l], dim=0) # [num_samples, hidden_dim]
        neg_tensor = torch.cat(activations["neg"][l], dim=0)
        
        pos_mean = pos_tensor.mean(dim=0)
        neg_mean = neg_tensor.mean(dim=0)
        
        vectors[l] = pos_mean - neg_mean
        
    return vectors