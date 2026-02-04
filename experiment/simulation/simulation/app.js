/**
 * app.js
 * Main application bootstrapping and event handling
 */

// Global instances
let networkRenderer;
let animationController;

// Application state
const appState = {
    currentStep: null,
    completedSteps: []
};

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
});

function initializeApp() {
    // Initialize network renderer
    networkRenderer = new NetworkRenderer('networkSVG', MLPData);

    // Initialize animation controller
    animationController = new AnimationController(networkRenderer, MLPData);

    // Update sample info display
    updateSampleInfo();

    // Initialize button states
    const step2 = document.getElementById('step2Btn');
    const step3 = document.getElementById('step3Btn');
    if (step2) step2.classList.add('disabled');
    if (step3) step3.classList.add('disabled');

    console.log('MLP Simulator initialized successfully');
}

function setupEventListeners() {
    // Step 1: Forward Pass
    const step1Btn = document.getElementById('step1Btn');
    step1Btn.addEventListener('click', async () => {
        if (animationController.isAnimating) return;

        setStepActive(1);
        step1Btn.classList.add('running');

        await animationController.animateForwardPass();

        step1Btn.classList.remove('running');
        step1Btn.classList.add('completed');
        step1Btn.classList.add('completed');
        appState.completedSteps.push(1);

        // Unlock Step 2
        const step2Btn = document.getElementById('step2Btn');
        if (step2Btn) step2Btn.classList.remove('disabled');
    });

    // Step 2: Backpropagation
    const step2Btn = document.getElementById('step2Btn');
    step2Btn.addEventListener('click', async () => {
        if (animationController.isAnimating) return;

        setStepActive(2);
        step2Btn.classList.add('running');

        await animationController.animateBackpropagation();

        step2Btn.classList.remove('running');
        step2Btn.classList.add('completed');
        step2Btn.classList.add('completed');
        appState.completedSteps.push(2);

        // Unlock Step 3
        const step3Btn = document.getElementById('step3Btn');
        if (step3Btn) step3Btn.classList.remove('disabled');
    });

    // Reset Button
    const resetSimBtn = document.getElementById('resetSimBtn');
    if (resetSimBtn) {
        resetSimBtn.addEventListener('click', () => {
            resetSimulation();
        });
    }

    // Step 3: Activation Function
    const step3Btn = document.getElementById('step3Btn');
    step3Btn.addEventListener('click', async () => {
        if (animationController.isAnimating || step3Btn.classList.contains('disabled')) return;

        setStepActive(3);
        step3Btn.classList.add('running');

        await animationController.animateActivationFunction();

        step3Btn.classList.remove('running');
        step3Btn.classList.add('completed');
        appState.completedSteps.push(3);

        // Lock all steps after completion
        lockAllSteps();

        // Show Reset Button
        if (resetSimBtn) resetSimBtn.style.display = 'flex';
    });

    // Configuration Inputs (Demonstration Only)
    const configInputs = ['actFuncSelect', 'optimizerSelect', 'epochsInput', 'lrInput'];
    configInputs.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('change', (e) => {
                const label = e.target.closest('.config-item').querySelector('label').innerText;
                const value = e.target.value;
                animationController.updateStatus(
                    `<strong>Config Change:</strong> ${label} set to ${value}.<br>` +
                    `<span style="font-size:0.8em; color:var(--text-muted)">(Note: Using pre-computed values for demonstration)</span>`
                );
            });
        }
    });
}

function lockAllSteps() {
    ['step1Btn', 'step2Btn', 'step3Btn'].forEach(id => {
        const btn = document.getElementById(id);
        if (btn) btn.classList.add('disabled');
    });
}

function resetSimulation() {
    // Reset App State
    appState.currentStep = null;
    appState.completedSteps = [];

    // Reset Animation Controller
    animationController.reset();

    // Reset Buttons UI
    const step1Btn = document.getElementById('step1Btn');
    const step2Btn = document.getElementById('step2Btn');
    const step3Btn = document.getElementById('step3Btn');
    const resetSimBtn = document.getElementById('resetSimBtn');

    if (step1Btn) {
        step1Btn.classList.remove('active', 'running', 'completed', 'disabled');
        step1Btn.querySelector('.step-num').textContent = '1';
    }

    if (step2Btn) {
        step2Btn.classList.remove('active', 'running', 'completed');
        step2Btn.classList.add('disabled');
        step2Btn.querySelector('.step-num').textContent = '2';
    }

    if (step3Btn) {
        step3Btn.classList.remove('active', 'running', 'completed');
        step3Btn.classList.add('disabled');
        step3Btn.querySelector('.step-num').textContent = '3';
    }

    // Hide Reset Button
    if (resetSimBtn) resetSimBtn.style.display = 'none';

    // Reset Visualization (Network)
    networkRenderer.resetNetwork();
}


function setStepActive(stepNum) {
    // Remove active class from all buttons
    const allButtons = document.querySelectorAll('.step-btn');
    allButtons.forEach(btn => {
        btn.classList.remove('active');
    });

    // Add active class to current button
    const currentBtn = document.getElementById(`step${stepNum}Btn`);
    if (currentBtn) {
        currentBtn.classList.add('active');
    }

    appState.currentStep = stepNum;
}


function updateSampleInfo() {
    const sampleLabel = document.getElementById('sampleLabel');
    if (sampleLabel) {
        sampleLabel.textContent = MLPData.sampleData.label;
    }
}

// Utility function for debugging
function logNetworkState() {
    console.log('Network Architecture:', MLPData.architecture);
    console.log('Sample Data:', MLPData.sampleData);
    console.log('Forward Pass Values:', MLPData.forwardPass);
    console.log('Backprop Gradients:', MLPData.backprop);
}

// Export for debugging
window.appState = appState;
window.networkRenderer = networkRenderer;
window.animationController = animationController;
window.MLPData = MLPData;
window.logNetworkState = logNetworkState;
