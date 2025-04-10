import torch
import matplotlib.pyplot as plt

# Create an instance of the model
model = BasicMLP()
hook_outputs = {}

# Define the hook function
def get_activation(name):
    def hook(module, input, output):
        hook_outputs[name] = output.detach().cpu().numpy()
    return hook

# Register the hooks for the FC layers
model.fc1.register_forward_hook(get_activation('fc1'))
model.fc2.register_forward_hook(get_activation('fc2'))

dummy_input = torch.randn(1, 28*28)
output = model(dummy_input)
print("Activations after fc1:", hook_outputs['fc1'].shape)
print("Activations after fc2:", hook_outputs['fc2'].shape)

def visualize_activations(activations, layer_name, num_neurons=10):
    """Visualizes the activations of a layer for a single input."""
    if layer_name in activations:
        layer_output = activations[layer_name]
        num_active_neurons = layer_output.shape[1]
        n_cols = min(num_neurons, num_active_neurons)
        plt.figure(figsize=(12, 2))
        for i in range(n_cols):
            plt.subplot(1, n_cols, i + 1)
            plt.hist(layer_output[0, i, :].flatten(), bins=20) # For Conv layers
            plt.title(f'Neuron {i+1}')
        plt.suptitle(f'Activations of {layer_name}')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Activations for {layer_name} not found.")

def visualize_fc_layer_output(activations, layer_name, num_neurons=10):
    """Visualizes the output of a fully connected layer."""
    if layer_name in activations:
        layer_output = activations[layer_name]
        num_active_neurons = layer_output.shape[1]
        n_cols = min(num_neurons, num_active_neurons)
        plt.figure(figsize=(12, 2))
        for i in range(n_cols):
            plt.subplot(1, n_cols, i + 1)
            plt.plot(layer_output[0, i, :].flatten()) # For Conv layers
            plt.title(f'Neuron {i+1}')
        plt.suptitle(f'Output of {layer_name}')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Activations for {layer_name} not found.")


# Visualize the activations
# visualize_activations(hook_outputs, 'fc1')
# visualize_activations(hook_outputs, 'fc2')

# If you want to track activations across multiple forward passes (e.g., for a batch):
all_fc1_activations = []
all_fc2_activations = []

# Perform multiple forward passes
num_samples_to_track = 5
for _ in range(num_samples_to_track):
    input_batch = torch.randn(1, 28*28)
    output = model(input_batch)
    all_fc1_activations.append(hook_outputs['fc1'])
    all_fc2_activations.append(hook_outputs['fc2'])

all_fc1_activations = np.concatenate(all_fc1_activations, axis=0)
all_fc2_activations = np.concatenate(all_fc2_activations, axis=0)

def visualize_neuron_distribution(activations, layer_name, neuron_index=0):
    """Visualizes the distribution of activations for a specific neuron across multiple samples."""
    if layer_name in hook_outputs:
        neuron_values = activations[:, neuron_index]
        plt.figure()
        plt.hist(neuron_values, bins=20)
        plt.title(f'Activation Distribution of Neuron {neuron_index} in {layer_name}')
        plt.xlabel('Activation Value')
        plt.ylabel('Frequency')
        plt.show()
    else:
        print(f"Activations for {layer_name} not found in hook_outputs.")

# Visualize the distribution of a single neuron's activations
visualize_neuron_distribution(all_fc1_activations, 'fc1', neuron_index=5)
visualize_neuron_distribution(all_fc2_activations, 'fc2', neuron_index=10)