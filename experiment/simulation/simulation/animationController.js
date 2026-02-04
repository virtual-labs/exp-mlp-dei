/**
 * animationController.js
 * Controls step-by-step animations for forward pass, backpropagation, and activation visualization
 */

class AnimationController {
    constructor(renderer, mlpData) {
        this.renderer = renderer;
        this.data = mlpData;
        this.isAnimating = false;
        this.currentStep = null;
    }

    async animateForwardPass() {
        if (this.isAnimating) return;

        this.isAnimating = true;
        this.currentStep = 'forward';
        this.renderer.resetNetwork();

        this.updateStatus('Starting Forward Pass...');
        await this.sleep(500);

        // Step 1: Show input layer
        this.updateStatus('Step 1: Input layer receives features [5.1, 3.5, 1.4, 0.2]');
        const inputActivations = this.data.forwardPass.layer0;
        for (let i = 0; i < inputActivations.length; i++) {
            this.renderer.highlightNeuron(0, i, '#0d6efd');
            this.renderer.showNeuronValue(0, i, inputActivations[i]);
            await this.sleep(200);
        }
        await this.sleep(800);

        // Step 2: Propagate to Hidden Layer 1
        this.updateStatus('Step 2: Computing Hidden Layer 1 (10 neurons with ReLU)');
        const h1Activations = this.data.forwardPass.layer1_a;

        for (let j = 0; j < h1Activations.length; j++) {
            // Animate connections from all input neurons to this hidden neuron
            for (let i = 0; i < inputActivations.length; i++) {
                this.renderer.highlightConnection(0, i, 1, j, '#3d8bfd');
                await this.sleep(30);
            }

            // Activate the neuron
            const color = h1Activations[j] > 0 ? '#27ae60' : '#e74c3c';
            this.renderer.highlightNeuron(1, j, color);
            this.renderer.showNeuronValue(1, j, h1Activations[j]);
            await this.sleep(150);
        }
        await this.sleep(800);

        // Step 3: Propagate to Hidden Layer 2
        this.updateStatus('Step 3: Computing Hidden Layer 2 (8 neurons with ReLU)');
        const h2Activations = this.data.forwardPass.layer2_a;

        for (let j = 0; j < h2Activations.length; j++) {
            for (let i = 0; i < h1Activations.length; i++) {
                this.renderer.highlightConnection(1, i, 2, j, '#3d8bfd');
                await this.sleep(25);
            }

            const color = h2Activations[j] > 0 ? '#27ae60' : '#e74c3c';
            this.renderer.highlightNeuron(2, j, color);
            this.renderer.showNeuronValue(2, j, h2Activations[j]);
            await this.sleep(150);
        }
        await this.sleep(800);

        // Step 4: Propagate to Output Layer
        this.updateStatus('Step 4: Computing Output Layer (3 neurons with Softmax)');
        const outputActivations = this.data.forwardPass.layer3_a;
        const classNames = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'];

        for (let j = 0; j < outputActivations.length; j++) {
            for (let i = 0; i < h2Activations.length; i++) {
                this.renderer.highlightConnection(2, i, 3, j, '#3d8bfd');
                await this.sleep(25);
            }

            this.renderer.highlightNeuron(3, j, '#6f42c1');
            this.renderer.showNeuronValue(3, j, outputActivations[j]);
            await this.sleep(200);
        }
        await this.sleep(1000);

        // Final result
        const predictedClass = outputActivations.indexOf(Math.max(...outputActivations));
        this.updateStatus(
            `✓ Forward Pass Complete!\\n\\n` +
            `Predicted: ${classNames[predictedClass]}\\n` +
            `Confidence: ${(outputActivations[predictedClass] * 100).toFixed(1)}%\\n\\n` +
            `Output probabilities:\\n` +
            `  ${classNames[0]}: ${(outputActivations[0] * 100).toFixed(1)}%\\n` +
            `  ${classNames[1]}: ${(outputActivations[1] * 100).toFixed(1)}%\\n` +
            `  ${classNames[2]}: ${(outputActivations[2] * 100).toFixed(1)}%`
        );

        this.isAnimating = false;
    }

    async animateBackpropagation() {
        if (this.isAnimating) return;

        this.isAnimating = true;
        this.currentStep = 'backprop';

        // First run forward pass to set up the network
        await this.animateForwardPass();
        await this.sleep(1000);

        this.updateStatus('Starting Backpropagation...');
        await this.sleep(500);

        // Step 1: Compute output layer error
        this.updateStatus('Step 1: Computing output layer error (Loss gradient)');
        const outputGrad = this.data.backprop.outputGrad;

        for (let i = 0; i < outputGrad.length; i++) {
            const color = outputGrad[i] < 0 ? '#27ae60' : '#e74c3c';
            this.renderer.highlightNeuron(3, i, color);
            await this.sleep(300);
        }
        await this.sleep(800);

        // Step 2: Backpropagate to Hidden Layer 2
        this.updateStatus('Step 2: Backpropagating gradients to Hidden Layer 2');
        const h2Grad = this.data.backprop.layer2_grad;

        for (let i = 0; i < h2Grad.length; i++) {
            // Show gradient flowing backward
            for (let j = 0; j < outputGrad.length; j++) {
                this.renderer.highlightConnection(2, i, 3, j, '#f39c12');
                await this.sleep(30);
            }

            if (h2Grad[i] > 0.001) {
                this.renderer.highlightNeuron(2, i, '#ffc107');
            }
            await this.sleep(150);
        }
        await this.sleep(800);

        // Step 3: Backpropagate to Hidden Layer 1
        this.updateStatus('Step 3: Backpropagating gradients to Hidden Layer 1');
        const h1Grad = this.data.backprop.layer1_grad;

        for (let i = 0; i < h1Grad.length; i++) {
            for (let j = 0; j < h2Grad.length; j++) {
                this.renderer.highlightConnection(1, i, 2, j, '#f39c12');
                await this.sleep(25);
            }

            if (h1Grad[i] > 0.001) {
                this.renderer.highlightNeuron(1, i, '#ffc107');
            }
            await this.sleep(150);
        }
        await this.sleep(800);

        // Step 4: Update weights
        this.updateStatus('Step 4: Updating weights using gradients');

        // Highlight all connections being updated
        const connections = this.renderer.svg.querySelectorAll('.connection');
        connections.forEach((conn, idx) => {
            setTimeout(() => {
                conn.setAttribute('stroke', '#27ae60');
                conn.setAttribute('stroke-width', '2');
            }, idx * 5);
        });

        await this.sleep(2000);

        this.updateStatus(
            `✓ Backpropagation Complete!\\n\\n` +
            `Gradients computed for all layers\\n` +
            `Weights updated using Adam optimizer\\n` +
            `Learning rate: 0.01\\n\\n` +
            `Average gradient magnitudes:\\n` +
            `  Layer 0→1: ${this.data.backprop.weightGrads.layer0to1.toFixed(4)}\\n` +
            `  Layer 1→2: ${this.data.backprop.weightGrads.layer1to2.toFixed(4)}\\n` +
            `  Layer 2→3: ${this.data.backprop.weightGrads.layer2to3.toFixed(4)}`
        );

        this.isAnimating = false;
    }

    async animateActivationFunction() {
        if (this.isAnimating) return;

        this.isAnimating = true;
        this.currentStep = 'activation';
        this.renderer.resetNetwork();

        // Show the activation plot
        const activationContainer = document.getElementById('activationContainer');
        if (activationContainer) activationContainer.style.display = 'block';

        // Get example neuron data for the plot
        const exampleData = this.data.activationFunction.exampleNeuron;
        const zValue = exampleData.z;
        const aValue = exampleData.a;

        // Brief, general explanation only
        this.updateStatus(
            `<strong>Applying ReLU Activation</strong><br>` +
            `Scanning all neurons...<br>` +
            `Rule: Output = max(0, Input)`
        );

        // Update activation info display
        const zEl = document.getElementById('zValue');
        const aEl = document.getElementById('aValue');
        if (zEl) zEl.textContent = zValue.toFixed(4);
        if (aEl) aEl.textContent = aValue.toFixed(4);

        // Draw the activation plot only if element exists
        const activationPlotEl = document.getElementById('activationPlot');
        if (activationPlotEl) {
            const plotDrawer = new ActivationPlotDrawer('activationPlot', this.data);
            plotDrawer.drawReLU(zValue, aValue);
        }

        await this.sleep(1000);

        // Process Hidden Layer 1
        const layer1_z = this.data.forwardPass.layer1_z;
        const layer1_a = this.data.forwardPass.layer1_a;

        for (let i = 0; i < layer1_z.length; i++) {
            const a = layer1_a[i];
            if (a > 0) {
                // Highlight activated neurons in green
                this.renderer.highlightNeuron(1, i, '#27ae60');
                this.renderer.showNeuronValue(1, i, a);
            } else {
                // Highlight deactivated neurons in red
                this.renderer.highlightNeuron(1, i, '#e74c3c');
                this.renderer.showNeuronValue(1, i, 0);
            }
            await this.sleep(50); // Fast iteration
        }

        // Process Hidden Layer 2
        const layer2_z = this.data.forwardPass.layer2_z;
        const layer2_a = this.data.forwardPass.layer2_a;

        for (let i = 0; i < layer2_z.length; i++) {
            const a = layer2_a[i];
            if (a > 0) {
                this.renderer.highlightNeuron(2, i, '#27ae60');
                this.renderer.showNeuronValue(2, i, a);
            } else {
                this.renderer.highlightNeuron(2, i, '#e74c3c');
                this.renderer.showNeuronValue(2, i, 0);
            }
            await this.sleep(50); // Fast iteration
        }

        await this.sleep(500);

        this.updateStatus(
            `<strong>Activation Visualization Complete</strong><br>` +
            `Non-linearity applied to network.`
        );

        this.isAnimating = false;
    }

    updateStatus(message) {
        const statusContent = document.getElementById('statusContent');
        if (statusContent) {
            statusContent.innerHTML = `<p>${message.replace(/\\n/g, '<br>')}</p>`;
        }
        // Log to console for debugging if status panel is removed
        console.log('Status:', message);
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    reset() {
        this.isAnimating = false;
        this.currentStep = null;
        this.renderer.resetNetwork();

        // Hide activation plot
        const activationContainer = document.getElementById('activationContainer');
        if (activationContainer) {
            activationContainer.style.display = 'none';
        }

        this.updateStatus('Click a step button to begin visualization');
    }
}
