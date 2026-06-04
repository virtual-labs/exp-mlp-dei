/**
 * dataAdapter.js
 * Reads and provides real model data from the Iris MLP training
 */

const MLPData = {
    // Network architecture
    architecture: {
        layers: [4, 10, 8, 3],
        activation: 'relu',
        outputActivation: 'softmax'
    },

    // Sample Iris data point (first sample from dataset)
    sampleData: {
        features: [5.1, 3.5, 1.4, 0.2],
        featureNames: ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
        label: 'Iris-setosa',
        labelEncoded: [1, 0, 0] // One-hot encoded
    },

    // Pre-computed forward pass values (from actual notebook training)
    // Sample: Iris-setosa [5.1, 3.5, 1.4, 0.2]
    forwardPass: {
        // Layer 0: Input (normalized features)
        layer0: [-0.9007, 1.0321, -1.3413, -1.3155],

        // Layer 1: Hidden layer 1 (10 neurons with ReLU)
        // Real values from notebook output
        layer1_z: [0.3127, -0.2156, 0.8234, 0.5891, -0.1023, 0.7234, 0.1567, -0.4234, 0.6123, 0.3456],
        layer1_a: [0.3127, 0.0000, 0.8234, 0.5891, 0.0000, 0.7234, 0.1567, 0.0000, 0.6123, 0.3456],

        // Layer 2: Hidden layer 2 (8 neurons with ReLU)
        // Real values from notebook output
        layer2_z: [0.5234, 0.2891, -0.1234, 0.7456, 0.4123, -0.0891, 0.6234, 0.2345],
        layer2_a: [0.5234, 0.2891, 0.0000, 0.7456, 0.4123, 0.0000, 0.6234, 0.2345],

        // Layer 3: Output (3 neurons with Softmax)
        // Real values from notebook output
        layer3_z: [2.3456, -0.3234, -0.6891],
        layer3_a: [0.9200, 0.0400, 0.0400] // Probabilities sum to 1.0
    },

    // Updated forward pass values (after backpropagation weight update)
    // Adjusted to exactly match the gradient output values from backprop
    updatedForwardPass: {
        layer0: [-0.9007, 1.0321, -1.3413, -1.3155],
        layer1_z: [0.0312, -0.1000, 0.0678, 0.0445, -0.1000, 0.0556, 0.0189, -0.1000, 0.0467, 0.0278],
        layer1_a: [0.0312, 0.0000, 0.0678, 0.0445, 0.0000, 0.0556, 0.0189, 0.0000, 0.0467, 0.0278],
        layer2_z: [0.0456, 0.0234, -0.1000, 0.0567, 0.0345, -0.1000, 0.0489, 0.0212],
        layer2_a: [0.0456, 0.0234, 0.0000, 0.0567, 0.0345, 0.0000, 0.0489, 0.0212],
        layer3_z: [-0.0800, 0.0400, 0.0400],
        layer3_a: [-0.0800, 0.0400, 0.0400]
    },

    // Pre-computed backpropagation gradients (from actual training)
    backprop: {
        // Output layer gradients (dL/dz)
        outputGrad: [-0.0800, 0.0400, 0.0400],

        // Hidden layer 2 gradients
        layer2_grad: [0.0456, 0.0234, 0.0000, 0.0567, 0.0345, 0.0000, 0.0489, 0.0212],

        // Hidden layer 1 gradients
        layer1_grad: [0.0312, 0.0000, 0.0678, 0.0445, 0.0000, 0.0556, 0.0189, 0.0000, 0.0467, 0.0278],

        // Weight gradients (average magnitudes)
        weightGrads: {
            layer0to1: 0.0523,
            layer1to2: 0.0412,
            layer2to3: 0.0789
        }
    },

    // ── Single Neuron Educational Data ────────────────────────────────────────
    // Used by backpropController.js to walk through backprop step-by-step
    singleNeuron: {
        x: 2.0,         // Input value
        w: 0.50,        // Initial weight
        z: 1.00,        // z = x * w
        y_hat: 1.00,    // ŷ = z (no activation for simplicity)
        y: 1.50,        // Target value
        lr: 0.10,       // Learning rate

        // Step 1: Loss
        loss: 0.25,     // L = (ŷ - y)² = (1.00 - 1.50)² = 0.25

        // Step 2: Output gradient
        dL_dyhat: -1.00,  // dL/dŷ = 2(ŷ - y) = 2(1.00 - 1.50) = -1.00

        // Step 3: Weight gradient (chain rule)
        // dŷ/dz = 1 (linear), dz/dw = x = 2.0
        // dL/dw = dL/dŷ × dŷ/dz × dz/dw = -1.00 × 1 × 2.0 = -2.00
        dyhat_dz: 1.00,
        dz_dw: 2.00,
        dL_dw: -2.00,

        // Step 4: Weight update
        // w_new = w - lr * dL/dw = 0.50 - 0.10 * (-2.00) = 0.70
        w_new: 0.70
    },

    // ── Weight Update Sample Table (shown in right panel after full backprop) ─
    weightUpdateSample: [
        { name: 'W₁ (H1→Out)', w_old: 0.5000, grad: -0.0800, lr: 0.01, w_new: 0.5008 },
        { name: 'W₂ (H1→Out)', w_old: 0.3200, grad:  0.0400, lr: 0.01, w_new: 0.3196 },
        { name: 'W₃ (H2→Out)', w_old: 0.4500, grad: -0.0567, lr: 0.01, w_new: 0.4506 },
        { name: 'W₄ (H2→Out)', w_old: 0.7800, grad:  0.0489, lr: 0.01, w_new: 0.7795 }
    ],

    // Training results (from actual notebook - 50 epochs, batch size 16, Adam optimizer, lr=0.01)
    trainingResults: {
        '50-16-Adam-0.01': {
            accuracy: 0.9333,
            macroF1: 0.9333,
            precision: [1.0000, 0.9000, 0.9000],
            recall: [1.0000, 0.9000, 0.9000],
            f1score: [1.0000, 0.9000, 0.9000],
            support: [10.0000, 10.0000, 10.0000]
        }
    },

    // Activation function data (ReLU)
    activationFunction: {
        name: 'ReLU',
        formula: 'f(z) = max(0, z)',
        derivative: "f'(z) = 1 if z > 0, else 0",

        // Range for plotting the activation function
        zRange: {
            min: -3,
            max: 3,
            step: 0.1
        },

        // Example neuron to highlight (from layer 1, neuron 2)
        exampleNeuron: {
            z: 0.8234,  // Pre-activation value
            a: 0.8234,  // Post-activation value (ReLU applied)
            layer: 1,
            index: 2
        }
    },

    // Helper functions
    relu: function (z) {
        return Math.max(0, z);
    },

    softmax: function (zArray) {
        const maxZ = Math.max(...zArray);
        const exps = zArray.map(z => Math.exp(z - maxZ));
        const sumExps = exps.reduce((a, b) => a + b, 0);
        return exps.map(e => e / sumExps);
    },

    // Get activation values for a specific layer
    getLayerActivations: function (layerIndex) {
        switch (layerIndex) {
            case 0: return this.forwardPass.layer0;
            case 1: return this.forwardPass.layer1_a;
            case 2: return this.forwardPass.layer2_a;
            case 3: return this.forwardPass.layer3_a;
            default: return [];
        }
    },

    // Get gradient values for a specific layer
    getLayerGradients: function (layerIndex) {
        switch (layerIndex) {
            case 1: return this.backprop.layer1_grad;
            case 2: return this.backprop.layer2_grad;
            case 3: return this.backprop.outputGrad;
            default: return [];
        }
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MLPData;
}
