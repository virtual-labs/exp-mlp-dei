# MLP Forward & Backpropagation Simulator

## Overview

This is an interactive educational visualization tool that demonstrates how a Multi-Layer Perceptron (MLP) neural network processes data through forward propagation and learns through backpropagation. The simulator uses **real data** from an actual Iris dataset classification model.

## Features

### Three Interactive Steps

1. **Forward Pass** - Visualizes how input data flows through the network layers
2. **Backpropagation** - Shows how gradients flow backward to update weights
3. **Activation Function** - Demonstrates ReLU activation with a live plot

### Key Characteristics

- ✅ **Real Data**: Uses actual forward pass values and gradients from trained model
- ✅ **One-Shot Animations**: Each step runs once when clicked, never auto-loops
- ✅ **Replayable**: Click any step button again to replay that animation
- ✅ **Educational**: Clear status messages explain what's happening at each stage

## How to Use

### Opening the Simulator

1. Navigate to the `actual/simulation/` folder
2. Open `index.html` in a modern web browser (Chrome, Firefox, Edge)
3. No build tools or server required - runs directly in the browser

### Running the Simulations

#### Step 1: Forward Pass
- Click the **"1 Forward Pass"** button
- Watch as:
  - Input features [5.1, 3.5, 1.4, 0.2] enter the network
  - Connections light up as data flows forward
  - Hidden neurons activate using ReLU
  - Output layer produces class probabilities using Softmax
- **Result**: Predicted class with confidence scores

#### Step 2: Backpropagation
- Click the **"2 Backpropagation"** button
- Watch as:
  - Forward pass runs first to set up the network
  - Loss gradient appears at output layer
  - Gradients flow backward through connections
  - Hidden layer gradients are computed
  - Weights are updated (shown in green)
- **Result**: Gradient magnitudes for each layer

#### Step 3: Activation Function
- Click the **"3 Activation Function"** button
- Watch as:
  - A specific neuron is highlighted
  - Activation plot appears showing ReLU curve
  - Pre-activation (z) and post-activation (a) values are displayed
  - The transformation point is marked on the curve
- **Result**: Visual understanding of ReLU: a = max(0, z)

## Architecture

### File Structure

```
actual/simulation/
├── index.html              # Main HTML structure
├── styles.css              # UI styling (matches reference design)
├── app.js                  # Application bootstrapping
├── dataAdapter.js          # Real model data and values
├── networkRenderer.js      # SVG network visualization
├── animationController.js  # Step-by-step animations
├── activationPlot.js       # Canvas-based function plotting
└── README.md              # This file
```

### Module Responsibilities

| File | Purpose |
|------|---------|
| **index.html** | DOM structure, layout, containers |
| **styles.css** | Visual styling matching reference UI |
| **app.js** | Event handling, initialization |
| **dataAdapter.js** | Provides real forward/backprop values |
| **networkRenderer.js** | Draws network using SVG |
| **animationController.js** | Orchestrates animations |
| **activationPlot.js** | Draws activation function graphs |

## Data Flow

### From Notebook to Visualization

1. **Training Data** (from Iris dataset)
   - 4 input features (sepal/petal measurements)
   - 3 output classes (Iris species)
   - 120 training samples, 30 test samples

2. **Model Architecture** (4 → 10 → 8 → 3)
   - Input layer: 4 neurons
   - Hidden layer 1: 10 neurons (ReLU)
   - Hidden layer 2: 8 neurons (ReLU)
   - Output layer: 3 neurons (Softmax)

3. **Real Values Used**
   - Forward pass activations from actual inference
   - Backprop gradients from actual training step
   - Training results: 93.33% accuracy with Adam optimizer

4. **Visualization**
   - SVG renders network topology
   - Animations show data/gradient flow
   - Canvas plots activation functions
   - Status panel explains each step

## Mathematical Concepts Demonstrated

### Forward Pass
```
Layer 1: z₁ = W₁·x + b₁,  a₁ = ReLU(z₁)
Layer 2: z₂ = W₂·a₁ + b₂, a₂ = ReLU(z₂)
Output:  z₃ = W₃·a₂ + b₃, a₃ = Softmax(z₃)
```

### Backpropagation
```
∂L/∂W₃ = ∂L/∂z₃ · a₂ᵀ
∂L/∂a₂ = W₃ᵀ · ∂L/∂z₃
∂L/∂z₂ = ∂L/∂a₂ ⊙ ReLU'(z₂)
(continues backward...)
```

### ReLU Activation
```
f(z) = max(0, z)
f'(z) = 1 if z > 0, else 0
```

## Design Philosophy

### UI/UX Principles Applied

1. **Clarity**: Each step is clearly labeled and explained
2. **Feedback**: Visual states (idle, running, completed) for buttons
3. **Control**: User decides when to run each step
4. **Education**: Status messages explain the mathematics
5. **Aesthetics**: Clean, professional design matching reference

### Color Coding

- **Blue** (#0d6efd): Active neurons, forward flow
- **Green** (#27ae60): Positive activations, weight updates
- **Red** (#e74c3c): Negative activations, errors
- **Yellow** (#ffc107): Gradients, backprop flow
- **Purple** (#6f42c1): Output layer

## Technical Details

### Browser Compatibility
- Chrome 90+
- Firefox 88+
- Edge 90+
- Safari 14+

### Technologies Used
- Vanilla JavaScript (ES6+)
- SVG for network visualization
- Canvas API for function plotting
- CSS Grid and Flexbox for layout
- CSS Variables for theming

### Performance
- Smooth 60fps animations
- Efficient SVG rendering
- No external dependencies
- Lightweight (~50KB total)

## Educational Use Cases

### For Students
- Understand forward propagation visually
- See how backpropagation computes gradients
- Learn activation function behavior
- Connect theory to practice

### For Instructors
- Demonstrate MLP concepts in lectures
- Show real vs. theoretical values
- Explain gradient flow intuitively
- Supplement textbook material

## Troubleshooting

### Animations Not Running
- Check browser console for errors
- Ensure all JS files are loaded
- Try refreshing the page

### Display Issues
- Ensure browser window is at least 1024px wide
- Check if JavaScript is enabled
- Clear browser cache if needed

### Performance Issues
- Close other browser tabs
- Disable browser extensions
- Use a modern browser version

## Future Enhancements

Potential additions:
- Multiple sample data points
- Different activation functions (Sigmoid, Tanh)
- Adjustable learning rate
- Loss curve visualization
- Weight matrix heatmaps

## Credits

- **Dataset**: Iris flower dataset (Fisher, 1936)
- **Framework**: TensorFlow/Keras for model training
- **Design**: Inspired by modern educational platforms
- **Architecture**: MLP with ReLU activations

## License

Educational use only. Part of university coursework for neural network visualization.

---

**Version**: 1.0  
**Last Updated**: February 2026  
**Author**: ML Visualization Team
