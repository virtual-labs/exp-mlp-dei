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

    // Show the persistent forward pass panel immediately (step cards visible throughout)
    this._showForwardPassPanel();
    this._setFpCardActive(0);
    await this.sleep(400);

    // Step 1: Show input layer
    const inputActivations = this.data.forwardPass.layer0;
    for (let i = 0; i < inputActivations.length; i++) {
      this.renderer.highlightNeuron(0, i, '#0d6efd');
      this.renderer.showNeuronValue(0, i, inputActivations[i]);
      await this.sleep(200);
    }
    await this.sleep(800);

    // Step 2: Propagate to Hidden Layer 1
    this._setFpCardActive(1);
    const h1Activations = this.data.forwardPass.layer1_a;

    for (let j = 0; j < h1Activations.length; j++) {
      for (let i = 0; i < inputActivations.length; i++) {
        this.renderer.highlightConnection(0, i, 1, j, '#3d8bfd');
        await this.sleep(30);
      }
      const color = h1Activations[j] > 0 ? '#27ae60' : '#adb5bd';
      this.renderer.highlightNeuron(1, j, color);
      this.renderer.showNeuronValue(1, j, h1Activations[j]);
      await this.sleep(150);
    }
    await this.sleep(800);

    // Step 3: Propagate to Hidden Layer 2
    this._setFpCardActive(2);
    const h2Activations = this.data.forwardPass.layer2_a;

    for (let j = 0; j < h2Activations.length; j++) {
      for (let i = 0; i < h1Activations.length; i++) {
        this.renderer.highlightConnection(1, i, 2, j, '#3d8bfd');
        await this.sleep(25);
      }
      const color = h2Activations[j] > 0 ? '#27ae60' : '#adb5bd';
      this.renderer.highlightNeuron(2, j, color);
      this.renderer.showNeuronValue(2, j, h2Activations[j]);
      await this.sleep(150);
    }
    await this.sleep(800);

    // Step 4: Propagate to Output Layer
    this._setFpCardActive(3);
    const outputActivations = this.data.forwardPass.layer3_a;

    for (let j = 0; j < outputActivations.length; j++) {
      for (let i = 0; i < h2Activations.length; i++) {
        this.renderer.highlightConnection(2, i, 3, j, '#3d8bfd');
        await this.sleep(25);
      }
      this.renderer.highlightNeuron(3, j, '#6f42c1');
      this.renderer.showNeuronValue(3, j, outputActivations[j]);
      await this.sleep(200);
    }
    await this.sleep(600);
    // Mark all cards done
    this._setFpCardsDone();

    this.isAnimating = false;
  }

  async animateBackpropagation() {
    if (this.isAnimating) return;
    // Delegate to the guided BackpropController
    const defaultExpl = document.getElementById('defaultExplanation');
    const bpPanel = document.getElementById('bpRightPanel');
    if (defaultExpl) defaultExpl.style.display = 'none';
    if (bpPanel) bpPanel.style.display = 'block';

    if (typeof bpController !== 'undefined' && bpController) {
      bpController.init();
    }
    // isAnimating stays false — user drives steps manually
  }

  async animateActivationFunction() {
    if (this.isAnimating) return;

    this.isAnimating = true;
    this.currentStep = 'activation';

    // Keep the main network SVG visible — do NOT switch to a separate view
    // Reset the network to a clean base first
    this.renderer.resetNetwork();

    // Hide bpView if it was showing (coming from step 2)
    const bpView = document.getElementById('bpView');
    if (bpView) bpView.style.display = 'none';
    const defaultViz = document.getElementById('defaultViz');
    if (defaultViz) defaultViz.style.display = 'block';

    // Switch right panel to activation panel
    ['defaultExplanation', 'forwardPassPanel', 'statusPanel', 'bpRightPanel'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.style.display = 'none';
    });
    const actRight = document.getElementById('activationRightPanel');
    if (actRight) actRight.style.display = 'block';

    // STATIC VISUALIZATION - No animation or delays

    // Show input layer (blue)
    const layer0 = this.data.updatedForwardPass.layer0;
    layer0.forEach((v, i) => {
      this.renderer.highlightNeuron(0, i, '#0d6efd');
      this.renderer.showNeuronValue(0, i, v);
    });

    // Hidden Layer 1 — colour by ReLU state, display a (which is z if >0, else 0)
    const layer1_z = this.data.updatedForwardPass.layer1_z;
    const layer1_a = this.data.updatedForwardPass.layer1_a;
    for (let i = 0; i < layer1_a.length; i++) {
      const z = layer1_z[i];
      const color = z > 0 ? '#27ae60' : '#adb5bd';
      this.renderer.highlightNeuron(1, i, color);
      this.renderer.showNeuronValue(1, i, z > 0 ? z : 0);
    }

    // Hidden Layer 2 — same treatment
    const layer2_z = this.data.updatedForwardPass.layer2_z;
    const layer2_a = this.data.updatedForwardPass.layer2_a;
    for (let i = 0; i < layer2_a.length; i++) {
      const z = layer2_z[i];
      const color = z > 0 ? '#27ae60' : '#adb5bd';
      this.renderer.highlightNeuron(2, i, color);
      this.renderer.showNeuronValue(2, i, z > 0 ? z : 0);
    }

    // Output layer (purple, show softmax probabilities)
    const layer3_a = this.data.updatedForwardPass.layer3_a;
    layer3_a.forEach((v, i) => {
      this.renderer.highlightNeuron(3, i, '#6f42c1');
      this.renderer.showNeuronValue(3, i, v);
    });

    this.isAnimating = false;
  }


  // ── Panel helpers ──────────────────────────────────────────────────────────

  _showStatusPanel() {
    ['defaultExplanation', 'forwardPassPanel', 'bpRightPanel', 'activationRightPanel'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.style.display = 'none';
    });
    const sp = document.getElementById('statusPanel');
    if (sp) sp.style.display = 'block';
  }

  _showForwardPassPanel() {
    ['defaultExplanation', 'statusPanel', 'bpRightPanel', 'activationRightPanel'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.style.display = 'none';
    });
    const fp = document.getElementById('forwardPassPanel');
    if (fp) fp.style.display = 'block';
  }

  // ── Forward Pass card helpers ───────────────────────────────────────────────
  _setFpCardActive(activeIdx) {
    for (let i = 0; i <= 3; i++) {
      const card = document.getElementById(`fpCard${i}`);
      if (!card) continue;
      card.classList.remove('fp-step-active', 'fp-step-done', 'fp-step-inactive');
      if (i < activeIdx) {
        card.classList.add('fp-step-done');
      } else if (i === activeIdx) {
        card.classList.add('fp-step-active');
      } else {
        card.classList.add('fp-step-inactive');
      }
    }
  }

  _setFpCardsDone() {
    for (let i = 0; i <= 3; i++) {
      const card = document.getElementById(`fpCard${i}`);
      if (!card) continue;
      card.classList.remove('fp-step-active', 'fp-step-done', 'fp-step-inactive');
      card.classList.add('fp-step-done');
    }
  }

  updateStatus(message) {
    const statusContent = document.getElementById('statusContent');
    if (statusContent) {
      statusContent.innerHTML = `<p>${message}</p>`;
    }
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  reset() {
    this.isAnimating = false;
    this.currentStep = null;
    this.renderer.resetNetwork();

    // Restore default network view
    const defaultViz = document.getElementById('defaultViz');
    const bpView = document.getElementById('bpView');
    const activationView = document.getElementById('activationView');
    if (defaultViz) defaultViz.style.display = 'block';
    if (bpView) bpView.style.display = 'none';
    if (activationView) activationView.style.display = 'none';
  }
}
