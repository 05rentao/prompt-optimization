import torch
from tqdm import tqdm

from .base import SteeringMethod


def _get_transformer_layers(model):
    """Return the decoder block list for common HF causal LM wrappers."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError("Could not resolve transformer layers on model (expected .model.layers or .layers).")


def _get_model_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("Model has no parameters; cannot infer device.") from exc


def _extract_hidden_states(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output:
        return output[0]
    if isinstance(output, list) and output:
        return output[0]
    raise TypeError(f"Unsupported layer output type for steering: {type(output)}")


class ActivationAddition(SteeringMethod):
    def apply(self, layer_idx: int, steering_vector: torch.Tensor, coefficient: float = 1.0):
        """Inject steering_vector at a decoder layer forward pass."""
        self.remove()

        layers = _get_transformer_layers(self.model)
        if layer_idx < 0 or layer_idx >= len(layers):
            raise IndexError(f"layer_idx={layer_idx} is out of range for {len(layers)} layers.")

        target_layer = layers[layer_idx]
        try:
            layer_param = next(target_layer.parameters())
        except StopIteration as exc:
            raise ValueError(f"Target layer {layer_idx} has no parameters; cannot infer dtype/device.") from exc

        config_hidden_size = getattr(getattr(self.model, "config", None), "hidden_size", None)
        hidden_size = int(config_hidden_size) if isinstance(config_hidden_size, int) and config_hidden_size > 0 else int(layer_param.shape[-1])
        vector = steering_vector.detach()
        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector.squeeze(0)
        elif vector.ndim != 1:
            raise ValueError(
                f"steering_vector must be shape [hidden_dim] or [1, hidden_dim], got {tuple(vector.shape)}"
            )
        if vector.shape[0] != hidden_size:
            raise ValueError(
                f"steering_vector hidden size mismatch: expected {hidden_size}, got {vector.shape[0]}"
            )
        vector = vector.to(device=layer_param.device)
        scaled_vector = coefficient * vector

        def steering_hook(module, inputs, output):
            hidden_states = _extract_hidden_states(output)
            steer = scaled_vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
            modified_hidden_states = hidden_states + steer
            if isinstance(output, torch.Tensor):
                return modified_hidden_states
            if isinstance(output, tuple):
                return (modified_hidden_states,) + output[1:]
            if isinstance(output, list):
                output[0] = modified_hidden_states
                return output
            raise TypeError(f"Unsupported layer output type for hook return: {type(output)}")

        self.handle = target_layer.register_forward_hook(steering_hook)
        print(f"[*] Steering applied to Layer {layer_idx} | Coeff: {coefficient}")


def format_qwen_prompt(tokenizer, instruction, response):
    """Formats prompts using Qwen's specific ChatML template."""
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def extract_steering_vectors(model, tokenizer, pos_data, neg_data, target_layers):
    """
    pos_data / neg_data: [{"instruction": "...", "response": "..."}]
    target_layers: layer indices to extract from.
    """
    if not pos_data or not neg_data:
        raise ValueError("pos_data and neg_data must both be non-empty.")
    if not target_layers:
        raise ValueError("target_layers must be a non-empty list of layer indices.")

    layers = _get_transformer_layers(model)
    for layer_idx in target_layers:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise IndexError(f"layer_idx={layer_idx} is out of range for {len(layers)} layers.")

    vectors = {}
    activations = {"pos": {l: [] for l in target_layers}, "neg": {l: [] for l in target_layers}}
    model_input_device = _get_model_input_device(model)

    def get_capture_hook(layer_idx, storage_dict, key):
        def hook(module, inputs, output):
            hidden_states = _extract_hidden_states(output)
            if hidden_states.ndim != 3:
                raise ValueError(
                    f"Expected hidden states shape [batch, seq, hidden], got {tuple(hidden_states.shape)}"
                )
            last_token_act = hidden_states[:, -1, :].detach().cpu()
            storage_dict[key][layer_idx].append(last_token_act)
            return None

        return hook

    print("Extracting positive activations...")
    pos_handles = []
    for layer_idx in target_layers:
        handle = layers[layer_idx].register_forward_hook(get_capture_hook(layer_idx, activations, "pos"))
        pos_handles.append(handle)

    for item in tqdm(pos_data):
        prompt = format_qwen_prompt(tokenizer, item["instruction"], item["response"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model_input_device)
        with torch.no_grad():
            model(**inputs)
    for handle in pos_handles:
        handle.remove()

    print("Extracting negative activations...")
    neg_handles = []
    for layer_idx in target_layers:
        handle = layers[layer_idx].register_forward_hook(get_capture_hook(layer_idx, activations, "neg"))
        neg_handles.append(handle)

    for item in tqdm(neg_data):
        prompt = format_qwen_prompt(tokenizer, item["instruction"], item["response"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model_input_device)
        with torch.no_grad():
            model(**inputs)
    for handle in neg_handles:
        handle.remove()

    for layer_idx in target_layers:
        if not activations["pos"][layer_idx]:
            raise ValueError(f"No positive activations were captured for layer {layer_idx}.")
        if not activations["neg"][layer_idx]:
            raise ValueError(f"No negative activations were captured for layer {layer_idx}.")
        pos_tensor = torch.cat(activations["pos"][layer_idx], dim=0)
        neg_tensor = torch.cat(activations["neg"][layer_idx], dim=0)
        vectors[layer_idx] = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)

    return vectors