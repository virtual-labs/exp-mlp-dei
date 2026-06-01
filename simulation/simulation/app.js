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
    networkRenderer = new NetworkRenderer('networkSVG', MLPData);
    animationController = new AnimationController(networkRenderer, MLPData);

    // Second renderer for the full-network panel inside #fnWrapper
    const renderer2 = new NetworkRenderer('networkSVG2', MLPData);
    bpController = new BackpropController(renderer2, MLPData);

    // Lock step 2 and 3 initially
    ['step2Btn', 'step3Btn'].forEach(id => {
        const btn = document.getElementById(id);
        if (btn) btn.classList.add('disabled');
    });

    console.log('MLP Simulator initialized');
}

function setupEventListeners() {

    // ── Step 1: Forward Pass ─────────────────────────────────────────────────
    const step1Btn = document.getElementById('step1Btn');
    step1Btn.addEventListener('click', async () => {
        if (animationController.isAnimating) return;
        setStepActive(1);
        step1Btn.classList.add('running');

        // Show status panel during animation (forwardPassPanel shown after)
        showRightPanel('status');

        await animationController.animateForwardPass();
        // animateForwardPass() switches to forwardPassPanel with real results when done

        // Populate the activation right panel calc values now that animation ran
        const exN = MLPData.activationFunction.exampleNeuron;
        const calcZ = document.getElementById('actCalcZ');
        const calcA = document.getElementById('actCalcA');
        const resA = document.getElementById('actResultA');
        if (calcZ) calcZ.textContent = exN.z.toFixed(4);
        if (calcA) calcA.textContent = exN.a.toFixed(4);
        if (resA) resA.textContent = exN.a.toFixed(4);

        step1Btn.classList.remove('running');
        step1Btn.classList.add('completed');
        appState.completedSteps.push(1);

        // Unlock Step 2
        const step2Btn = document.getElementById('step2Btn');
        if (step2Btn) step2Btn.classList.remove('disabled');

        // Show Reset button
        const resetSimBtn = document.getElementById('resetSimBtn');
        if (resetSimBtn) resetSimBtn.style.display = 'flex';
    });

    // ── Step 2: Backpropagation ──────────────────────────────────────────────
    const step2Btn = document.getElementById('step2Btn');
    step2Btn.addEventListener('click', async () => {
        if (animationController.isAnimating || step2Btn.classList.contains('disabled')) return;
        setStepActive(2);
        step2Btn.classList.add('running');

        // Show vertical step list, switch right panel
        const vertSteps = document.getElementById('bpVertSteps');
        if (vertSteps) vertSteps.style.display = 'block';
        showRightPanel('backprop');

        await animationController.animateBackpropagation();

        step2Btn.classList.remove('running');
        step2Btn.classList.add('completed');
        appState.completedSteps.push(2);

        // Unlock Step 3
        const step3Btn = document.getElementById('step3Btn');
        if (step3Btn) step3Btn.classList.remove('disabled');

        const resetSimBtn = document.getElementById('resetSimBtn');
        if (resetSimBtn) resetSimBtn.style.display = 'flex';
    });

    // ── Reset ────────────────────────────────────────────────────────────────
    const resetSimBtn = document.getElementById('resetSimBtn');
    if (resetSimBtn) {
        resetSimBtn.addEventListener('click', () => resetSimulation());
    }

    // ── Step 3: Activation Function ──────────────────────────────────────────
    const step3Btn = document.getElementById('step3Btn');
    step3Btn.addEventListener('click', async () => {
        if (animationController.isAnimating || step3Btn.classList.contains('disabled')) return;
        setStepActive(3);
        step3Btn.classList.add('running');

        // Switch to activation right panel before animation
        showRightPanel('activation');

        await animationController.animateActivationFunction();

        step3Btn.classList.remove('running');
        step3Btn.classList.add('completed');
        appState.completedSteps.push(3);

        lockAllSteps();

        const resetBtnEl = document.getElementById('resetSimBtn');
        if (resetBtnEl) resetBtnEl.style.display = 'flex';
    });
}

// ── Panel Switcher ────────────────────────────────────────────────────────────
function showRightPanel(mode) {
    // Hide all right panels
    ['defaultExplanation', 'forwardPassPanel', 'statusPanel', 'bpRightPanel', 'activationRightPanel'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = 'none';
    });

    if (mode === 'forward') {
        const fp = document.getElementById('forwardPassPanel');
        if (fp) fp.style.display = 'block';
    } else if (mode === 'status') {
        const sp = document.getElementById('statusPanel');
        if (sp) sp.style.display = 'block';
    } else if (mode === 'backprop') {
        const bpPanel = document.getElementById('bpRightPanel');
        if (bpPanel) bpPanel.style.display = 'block';
    } else if (mode === 'activation') {
        const actPanel = document.getElementById('activationRightPanel');
        if (actPanel) actPanel.style.display = 'block';
    } else {
        const def = document.getElementById('defaultExplanation');
        if (def) def.style.display = 'block';
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function lockAllSteps() {
    ['step1Btn', 'step2Btn', 'step3Btn'].forEach(id => {
        const btn = document.getElementById(id);
        if (btn) btn.classList.add('disabled');
    });
}

function resetSimulation() {
    appState.currentStep = null;
    appState.completedSteps = [];

    animationController.reset();
    if (bpController) bpController.reset();

    // Hide vertical step list
    const vertSteps = document.getElementById('bpVertSteps');
    if (vertSteps) vertSteps.style.display = 'none';

    // Restore default right panel
    showRightPanel('default');

    // Reset buttons
    ['step1Btn', 'step2Btn', 'step3Btn'].forEach((id, i) => {
        const btn = document.getElementById(id);
        if (!btn) return;
        btn.classList.remove('active', 'running', 'completed', 'disabled');
        if (i > 0) btn.classList.add('disabled');
        const numEl = btn.querySelector('.step-num');
        if (numEl) numEl.textContent = String(i + 1);
    });

    const resetSimBtn = document.getElementById('resetSimBtn');
    if (resetSimBtn) resetSimBtn.style.display = 'none';

    networkRenderer.resetNetwork();
}

function setStepActive(stepNum) {
    document.querySelectorAll('.step-btn').forEach(btn => btn.classList.remove('active'));
    const currentBtn = document.getElementById(`step${stepNum}Btn`);
    if (currentBtn) currentBtn.classList.add('active');
    appState.currentStep = stepNum;
}
