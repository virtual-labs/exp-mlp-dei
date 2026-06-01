/**
 * backpropController.js
 * Guides the educational Backpropagation experience.
 * Steps: 0=Setup, 1=Loss, 2=OutputGrad, 3=WeightGrad, 4=WeightUpdate
 */

class BackpropController {
    constructor(renderer, mlpData) {
        this.renderer = renderer;
        this.data = mlpData;
        this.sn = mlpData.singleNeuron;
        this.currentStep = -1;
        this.isFullNetwork = false;
    }

    // ─── PUBLIC API ────────────────────────────────────────────────────────────

    init() {
        this.currentStep = 0;
        this.isFullNetwork = false;
        this._showBpView(true);
        this._renderSnDiagram();
        this.showStep(0);
        this._updateVerticalStepList(0);
    }

    showStep(n) {
        this.currentStep = n;
        this._updateVerticalStepList(n);
        this._updateRightPanel(n);
        this._updateSnDiagram(n);
    }

    nextStep() {
        const next = this.currentStep + 1;
        if (next <= 4) this.showStep(next);
    }

    async switchToFullNetwork() {
        this.isFullNetwork = true;
        const snWrapper = document.getElementById('snWrapper');
        const fnWrapper = document.getElementById('fnWrapper');
        const switchBtn = document.getElementById('bpSwitchBtn');
        if (snWrapper) snWrapper.style.display = 'none';
        if (fnWrapper) { fnWrapper.style.display = 'flex'; fnWrapper.style.flexDirection = 'column'; }
        if (switchBtn) switchBtn.style.display = 'none';
        this._updateRightPanelFullNetwork();
        await this._animateFullNetworkBackprop();
        this._showActivationNavButton();
    }

    reset() {
        this.currentStep = -1;
        this.isFullNetwork = false;
        this._showBpView(false);
        const fnWrapper = document.getElementById('fnWrapper');
        const snWrapper = document.getElementById('snWrapper');
        if (fnWrapper) fnWrapper.style.display = 'none';
        if (snWrapper) snWrapper.style.display = 'block';
        const switchBtn = document.getElementById('bpSwitchBtn');
        if (switchBtn) switchBtn.style.display = 'block';
        const rp = document.getElementById('bpRightPanel');
        if (rp) rp.innerHTML = '';
        this._updateVerticalStepList(-1);
    }

    // ─── VERTICAL STEP LIST ────────────────────────────────────────────────────

    _updateVerticalStepList(activeStep) {
        const steps = [
            { id: 'bpvs-0', num: '0', label: 'Setup' },
            { id: 'bpvs-1', num: '1', label: 'Loss' },
            { id: 'bpvs-2', num: '2', label: 'Output Gradient' },
            { id: 'bpvs-3', num: '3', label: 'Weight Gradient' },
            { id: 'bpvs-4', num: '4', label: 'Weight Update' }
        ];
        steps.forEach((s, i) => {
            const el = document.getElementById(s.id);
            if (!el) return;
            el.classList.remove('bpvs-active', 'bpvs-done', 'bpvs-pending');
            if (activeStep < 0) {
                el.classList.add('bpvs-pending');
            } else if (i < activeStep) {
                el.classList.add('bpvs-done');
                el.querySelector('.bpvs-num').textContent = '✓';
            } else if (i === activeStep) {
                el.classList.add('bpvs-active');
                el.querySelector('.bpvs-num').textContent = s.num;
            } else {
                el.classList.add('bpvs-pending');
                el.querySelector('.bpvs-num').textContent = s.num;
            }
        });
    }

    // ─── RIGHT PANEL ──────────────────────────────────────────────────────────

    _updateRightPanel(step) {
        const rp = document.getElementById('bpRightPanel');
        if (!rp) return;
        const sn = this.sn;

        // Pre-compute updated loss for step 4
        const y_hat_new = sn.x * sn.w_new;
        const loss_new = Math.pow(y_hat_new - sn.y, 2).toFixed(4);

        const configs = {
            0: {
                title: 'Initial Setup',
                badge: 'Setup', badgeColor: '#6c757d',
                formula: `z &nbsp;= x × w<br>ŷ &nbsp;= z &nbsp;(linear output)<br>η &nbsp;= learning rate`,
                calc:
                    `x &nbsp;= ${sn.x}<br>` +
                    `w &nbsp;= ${sn.w}<br>` +
                    `z &nbsp;= x × w = ${sn.x} × ${sn.w} = <strong>${sn.z.toFixed(2)}</strong><br>` +
                    `ŷ &nbsp;= <strong>${sn.y_hat.toFixed(2)}</strong><br>` +
                    `y (target) = <strong>${sn.y}</strong>`,
                result: `Initial weight: <strong>w = ${sn.w}</strong><br>Learning rate: <strong>η = ${sn.lr}</strong>`,
                explanation:
                    `We have an input <strong>x = ${sn.x}</strong> and a weight <strong>w = ${sn.w}</strong>. ` +
                    `The neuron multiplies them to get output <strong>ŷ = ${sn.y_hat.toFixed(2)}</strong>, ` +
                    `but the target is <strong>y = ${sn.y}</strong>. The network is wrong — let's fix it step by step.`
            },
            1: {
                title: 'Step 1: Compute the Loss',
                badge: 'Loss', badgeColor: '#dc3545',
                formula:
                    `<strong>Loss Function: Mean Squared Error (MSE)</strong><br>` +
                    `L = (ŷ − y)²`,
                calc:
                    `Predicted (ŷ) = ${sn.y_hat.toFixed(2)}<br>` +
                    `Target (y) &nbsp;= ${sn.y.toFixed(2)}<br>` +
                    `Difference &nbsp;= ŷ − y = ${(sn.y_hat - sn.y).toFixed(2)}<br>` +
                    `L &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= (${(sn.y_hat - sn.y).toFixed(2)})² = <strong>${sn.loss.toFixed(4)}</strong>`,
                result: `Loss = <strong style="color:#dc3545">${sn.loss.toFixed(4)}</strong>`,
                explanation:
                    `MSE measures how wrong our prediction is. Squaring the difference makes all errors positive ` +
                    `and penalizes larger mistakes more. We want to drive this loss toward zero.`
            },
            2: {
                title: 'Step 2: Output Gradient',
                badge: 'Output Grad', badgeColor: '#fd7e14',
                formula:
                    `<strong>Loss used: MSE — L = (ŷ − y)²</strong><br>` +
                    `Gradient of L with respect to ŷ:<br>` +
                    `dL/dŷ = 2 × (ŷ − y)`,
                calc:
                    `dL/dŷ = 2 × (${sn.y_hat.toFixed(2)} − ${sn.y.toFixed(2)})<br>` +
                    `&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= 2 × (${(sn.y_hat - sn.y).toFixed(2)})<br>` +
                    `&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <strong>${sn.dL_dyhat.toFixed(4)}</strong>`,
                result: `Output Gradient = <strong style="color:#fd7e14">${sn.dL_dyhat.toFixed(4)}</strong>`,
                explanation:
                    `This tells us: "Which direction and how much should the output change to reduce the loss?" ` +
                    `A <strong>negative gradient (${sn.dL_dyhat.toFixed(2)})</strong> means the output ` +
                    `needs to <em>increase</em> to match the target.`
            },
            3: {
                title: 'Step 3: Weight Gradient (Chain Rule)',
                badge: 'Weight Grad', badgeColor: '#fd7e14',
                formula:
                    `<strong>Why the chain rule?</strong><br>` +
                    `The weight <em>w</em> doesn't directly affect the loss. It affects <em>z</em>, which affects <em>ŷ</em>, which affects <em>L</em>. ` +
                    `We trace this chain of effects backward.<br><br>` +
                    `dL/dw = dL/dŷ × dŷ/dz × dz/dw`,
                calc:
                    `dŷ/dz = 1 &nbsp;(linear neuron, no activation)<br>` +
                    `dz/dw = x = ${sn.dz_dw}<br>` +
                    `dL/dw = ${sn.dL_dyhat.toFixed(2)} × ${sn.dyhat_dz.toFixed(2)} × ${sn.dz_dw.toFixed(2)}<br>` +
                    `&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <strong>${sn.dL_dw.toFixed(4)}</strong>`,
                result: `Weight Gradient = <strong style="color:#fd7e14">${sn.dL_dw.toFixed(4)}</strong>`,
                explanation:
                    `The weight gradient <strong>${sn.dL_dw.toFixed(2)}</strong> tells us: ` +
                    `"For every unit increase in this weight, the loss changes by ${sn.dL_dw.toFixed(2)}." ` +
                    `A negative gradient means increasing the weight reduces the loss, so we should increase it.`,
                extra: `<div class="bp-chain-rule">
                    <span>dL/dŷ</span><span>×</span><span>dŷ/dz</span><span>×</span><span>dz/dw</span>
                    <br>
                    <span class="bp-chain-val">${sn.dL_dyhat}</span><span>×</span><span class="bp-chain-val">${sn.dyhat_dz}</span><span>×</span><span class="bp-chain-val">${sn.dz_dw}</span>
                </div>`
            },
            4: {
                title: 'Step 4: Weight Update',
                badge: 'Update', badgeColor: '#198754',
                formula: `w_new = w_old − η × dL/dw`,
                calc:
                    `w_new = ${sn.w.toFixed(2)} − ${sn.lr.toFixed(2)} × (${sn.dL_dw.toFixed(2)})<br>` +
                    `&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= ${sn.w.toFixed(2)} + ${(sn.lr * Math.abs(sn.dL_dw)).toFixed(2)}<br>` +
                    `&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <strong>${sn.w_new.toFixed(4)}</strong>`,
                result:
                    `w: <span style="text-decoration:line-through;color:#dc3545">${sn.w}</span> → <strong style="color:#198754">${sn.w_new}</strong><br>` +
                    `New ŷ = ${sn.x} × ${sn.w_new} = <strong>${y_hat_new.toFixed(4)}</strong><br>` +
                    `New Loss = <strong style="color:#198754">${loss_new}</strong> &nbsp;↓ <span style="color:#888;font-size:0.85rem">(was ${sn.loss.toFixed(4)})</span>`,
                explanation:
                    `The gradient tells us how much this weight contributed to the error and in which direction. ` +
                    `We subtract a small fraction (learning rate × gradient) so the weight moves in the direction that reduces error. ` +
                    `With <strong>w = ${sn.w_new.toFixed(2)}</strong>, the new loss is <strong>${loss_new}</strong> — lower than before!`
            }
        };

        const c = configs[step];
        if (!c) return;

        rp.innerHTML = `
            <div class="bp-step-header">
                <span class="bp-step-badge" style="background:${c.badgeColor}">${c.badge}</span>
                <h3 class="bp-step-title">${c.title}</h3>
            </div>
            <div class="bp-formula-block">
                <div class="bp-section-label">Formula</div>
                <div class="bp-formula-text">${c.formula}</div>
            </div>
            <div class="bp-value-block">
                <div class="bp-section-label">Calculation</div>
                <div class="bp-value-text">${c.calc}</div>
            </div>
            <div class="bp-result-block">
                <div class="bp-section-label">Result</div>
                <div class="bp-result-text">${c.result}</div>
            </div>
            ${c.extra ? `<div class="bp-extra-block">${c.extra}</div>` : ''}
            <div class="bp-explanation-block">
                <div class="bp-section-label">What This Means</div>
                <p class="bp-expl-text">${c.explanation}</p>
            </div>
            ${step < 4
                ? `<button class="bp-next-btn" onclick="bpController.nextStep()">Next Step &rarr;</button>`
                : `<button class="bp-next-btn bp-full-net-btn" onclick="bpController.switchToFullNetwork()">Apply to Full Network &rarr;</button>`
            }
        `;
    }

    _updateRightPanelFullNetwork() {
        const rp = document.getElementById('bpRightPanel');
        if (!rp) return;
        rp.innerHTML = `
            <div class="bp-step-header">
                <span class="bp-step-badge" style="background:#6f42c1">Full Network</span>
                <h3 class="bp-step-title">Backprop Across All Layers</h3>
            </div>
            <div class="bp-explanation-block">
                <div class="bp-section-label">Same Concept, Scaled Up</div>
                <p class="bp-expl-text">The same 4 steps you just learned happen simultaneously for <em>every weight</em> in the network. Each connection gets its own gradient via the chain rule, and all weights are updated together.</p>
            </div>
            <div class="bp-formula-block">
                <div class="bp-section-label">Color Key</div>
                <div style="font-size:0.9rem; line-height:2.2;">
                    <span style="color:#0d6efd">●</span> <strong>Blue</strong> — Forward signal<br>
                    <span style="color:#fd7e14">●</span> <strong>Orange</strong> — Gradient flowing back<br>
                    <span style="color:#198754">●</span> <strong>Green</strong> — Updated weight<br>
                    <span style="color:#adb5bd">●</span> <strong>Grey</strong> — Inactive (ReLU = 0)
                </div>
            </div>
            <div id="activationNavBtn" style="display:none; margin-top:16px;">
                <button class="bp-next-btn" style="background:#6f42c1;" onclick="goToActivationFunction()">
                    Next: Activation Function →
                </button>
            </div>
        `;
    }

    _showActivationNavButton() {
        const btn = document.getElementById('activationNavBtn');
        if (btn) btn.style.display = 'block';
    }

    // ─── SINGLE NEURON DIAGRAM ─────────────────────────────────────────────────

    _renderSnDiagram() {
        const container = document.getElementById('snDiagram');
        if (!container) return;
        const sn = this.sn;
        container.innerHTML = `
            <div class="snv-pipeline" id="snv-pipeline">

                <div class="snv-block snv-block-input" id="snv-input">
                    <div class="snv-label">INPUT (x)</div>
                    <div class="snv-value">${sn.x}</div>
                </div>
                <div class="snv-arrow" id="snv-arr-1">↓</div>

                <div class="snv-block snv-block-weight" id="snv-weight">
                    <div class="snv-label">WEIGHT (w)</div>
                    <div class="snv-value" id="snv-w-val">w = <strong>${sn.w}</strong></div>
                    <div class="snv-sub">multiplied with input</div>
                </div>
                <div class="snv-arrow" id="snv-arr-2">↓</div>

                <div class="snv-block snv-block-neuron" id="snv-neuron">
                    <div class="snv-label">NEURON (z)</div>
                    <div class="snv-value" id="snv-z-val">z = x · w = <strong>${sn.z.toFixed(2)}</strong></div>
                    <div class="snv-sub">weighted sum</div>
                </div>
                <div class="snv-arrow" id="snv-arr-3">↓</div>

                <div class="snv-block snv-block-output" id="snv-output">
                    <div class="snv-label">OUTPUT (ŷ)</div>
                    <div class="snv-value" id="snv-yhat-val">ŷ = <strong>${sn.y_hat.toFixed(2)}</strong></div>
                    <div class="snv-sub">model prediction</div>
                </div>
                <div class="snv-arrow" id="snv-arr-4">↓</div>

                <div class="snv-block snv-block-target" id="snv-target">
                    <div class="snv-label">TARGET (y)</div>
                    <div class="snv-value">${sn.y}</div>
                    <div class="snv-sub">correct answer</div>
                </div>
                <div class="snv-arrow" id="snv-arr-5">↓</div>

                <div class="snv-block snv-block-loss" id="snv-loss">
                    <div class="snv-label">LOSS</div>
                    <div class="snv-value" id="snv-loss-val">L = (ŷ − y)² = <strong>${sn.loss.toFixed(4)}</strong></div>
                    <div class="snv-sub">how wrong we are</div>
                </div>
                <div class="snv-arrow" id="snv-arr-6">↓</div>

                <div class="snv-block snv-block-grad" id="snv-grad">
                    <div class="snv-label">GRADIENT</div>
                    <div class="snv-value" id="snv-grad-val">dL/dw = <strong>${sn.dL_dw.toFixed(4)}</strong></div>
                    <div class="snv-sub">direction of correction</div>
                </div>
                <div class="snv-arrow" id="snv-arr-7">↓</div>

                <div class="snv-block snv-block-updated" id="snv-updated">
                    <div class="snv-label">UPDATED WEIGHT</div>
                    <div class="snv-value" id="snv-wnew-val">
                        w_new = <span class="snv-old">${sn.w}</span> → <strong class="snv-new">${sn.w_new}</strong>
                    </div>
                    <div class="snv-sub">w − η × dL/dw</div>
                </div>

            </div>
        `;
    }

    _updateSnDiagram(step) {
        const sn = this.sn;

        const allBlocks = ['snv-input','snv-weight','snv-neuron','snv-output','snv-target','snv-loss','snv-grad','snv-updated'];
        const allArrows = ['snv-arr-1','snv-arr-2','snv-arr-3','snv-arr-4','snv-arr-5','snv-arr-6','snv-arr-7'];

        // Reset all to faded
        allBlocks.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.classList.remove('snv-active','snv-done','snv-error','snv-success','snv-highlight-grad');
                el.classList.add('snv-faded');
            }
        });
        allArrows.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.classList.add('snv-faded');
        });

        const activate = (id, cls = 'snv-active') => {
            const el = document.getElementById(id);
            if (el) { el.classList.remove('snv-faded'); el.classList.add(cls); }
        };
        const showArrow = id => {
            const el = document.getElementById(id);
            if (el) el.classList.remove('snv-faded');
        };
        const done = id => {
            const el = document.getElementById(id);
            if (el) { el.classList.remove('snv-faded'); el.classList.add('snv-done'); }
        };

        if (step === 0) {
            activate('snv-input');  showArrow('snv-arr-1');
            activate('snv-weight'); showArrow('snv-arr-2');
            activate('snv-neuron'); showArrow('snv-arr-3');
            activate('snv-output');
        }
        else if (step === 1) {
            done('snv-input');  showArrow('snv-arr-1');
            done('snv-weight'); showArrow('snv-arr-2');
            done('snv-neuron'); showArrow('snv-arr-3');
            activate('snv-output','snv-error'); showArrow('snv-arr-4');
            activate('snv-target');             showArrow('snv-arr-5');
            activate('snv-loss');
            const lossEl = document.getElementById('snv-loss-val');
            if (lossEl) lossEl.innerHTML = `L = (ŷ − y)² = <strong>${sn.loss.toFixed(4)}</strong>`;
        }
        else if (step === 2) {
            done('snv-input');  showArrow('snv-arr-1');
            done('snv-weight'); showArrow('snv-arr-2');
            done('snv-neuron'); showArrow('snv-arr-3');
            done('snv-output'); showArrow('snv-arr-4');
            done('snv-target'); showArrow('snv-arr-5');
            done('snv-loss');   showArrow('snv-arr-6');
            activate('snv-grad','snv-highlight-grad');
            const gradEl = document.getElementById('snv-grad-val');
            if (gradEl) gradEl.innerHTML = `dL/dŷ = <strong>${sn.dL_dyhat.toFixed(4)}</strong>`;
        }
        else if (step === 3) {
            done('snv-input');  showArrow('snv-arr-1');
            activate('snv-weight','snv-highlight-grad'); showArrow('snv-arr-2');
            done('snv-neuron'); showArrow('snv-arr-3');
            done('snv-output'); showArrow('snv-arr-4');
            done('snv-target'); showArrow('snv-arr-5');
            done('snv-loss');   showArrow('snv-arr-6');
            activate('snv-grad','snv-highlight-grad');
            const gradEl = document.getElementById('snv-grad-val');
            if (gradEl) gradEl.innerHTML = `dL/dw = <strong>${sn.dL_dw.toFixed(4)}</strong>`;
        }
        else if (step === 4) {
            const y_hat_new = sn.x * sn.w_new;
            const loss_new  = Math.pow(y_hat_new - sn.y, 2).toFixed(4);
            done('snv-input');  showArrow('snv-arr-1');
            done('snv-weight'); showArrow('snv-arr-2');
            done('snv-neuron'); showArrow('snv-arr-3');
            done('snv-output'); showArrow('snv-arr-4');
            done('snv-target'); showArrow('snv-arr-5');
            done('snv-loss');   showArrow('snv-arr-6');
            done('snv-grad');   showArrow('snv-arr-7');
            activate('snv-updated','snv-success');
            const zEl = document.getElementById('snv-z-val');
            if (zEl) zEl.innerHTML = `z = x · w_new = <strong>${(sn.x * sn.w_new).toFixed(4)}</strong>`;
            const lossEl = document.getElementById('snv-loss-val');
            if (lossEl) lossEl.innerHTML = `New Loss = <strong style="color:#198754">${loss_new}</strong> ↓`;
            const wnewEl = document.getElementById('snv-wnew-val');
            if (wnewEl) wnewEl.innerHTML =
                `w_new = <span class="snv-old">${sn.w}</span> → <strong class="snv-new">${sn.w_new}</strong>`;
        }
    }

    // ─── FULL NETWORK ANIMATION ────────────────────────────────────────────────

    async _animateFullNetworkBackprop() {
        const sleep = (ms) => new Promise(r => setTimeout(r, ms));
        const outputGrad = this.data.backprop.outputGrad;
        const h2Grad = this.data.backprop.layer2_grad;
        const h1Grad = this.data.backprop.layer1_grad;

        this._showNodeValues();
        await sleep(800);

        for (let i = 0; i < outputGrad.length; i++) {
            const color = '#6f42c1'; // Keep output neuron purple
            this.renderer.highlightNeuron(3, i, color);
            this.renderer.showNeuronValue(3, i, outputGrad[i]);
            await sleep(300);
        }
        await sleep(600);

        for (let i = 0; i < h2Grad.length; i++) {
            for (let j = 0; j < outputGrad.length; j++) {
                this.renderer.highlightConnection(2, i, 3, j, '#fd7e14');
                await sleep(20);
            }
            if (h2Grad[i] > 0.001) {
                this.renderer.highlightNeuron(2, i, '#fd7e14');
                this.renderer.showNeuronValue(2, i, h2Grad[i]);
            }
            await sleep(100);
        }
        await sleep(600);

        for (let i = 0; i < h1Grad.length; i++) {
            for (let j = 0; j < h2Grad.length; j++) {
                this.renderer.highlightConnection(1, i, 2, j, '#fd7e14');
                await sleep(15);
            }
            if (h1Grad[i] > 0.001) {
                this.renderer.highlightNeuron(1, i, '#fd7e14');
                this.renderer.showNeuronValue(1, i, h1Grad[i]);
            }
            await sleep(80);
        }
        await sleep(600);

        // Update all weights — green
        const connections = this.renderer.svg.querySelectorAll('.connection');
        connections.forEach((conn, idx) => {
            setTimeout(() => {
                conn.setAttribute('stroke', '#198754');
                conn.setAttribute('stroke-width', '2.5');
            }, idx * 4);
        });
        await sleep(1500);
    }

    _showNodeValues() {
        const fp = this.data.forwardPass;
        fp.layer0.forEach((v, i) => { this.renderer.highlightNeuron(0, i, '#0d6efd'); this.renderer.showNeuronValue(0, i, v); });
        fp.layer1_a.forEach((v, i) => { this.renderer.highlightNeuron(1, i, v > 0 ? '#27ae60' : '#adb5bd'); this.renderer.showNeuronValue(1, i, v); });
        fp.layer2_a.forEach((v, i) => { this.renderer.highlightNeuron(2, i, v > 0 ? '#27ae60' : '#adb5bd'); this.renderer.showNeuronValue(2, i, v); });
        fp.layer3_a.forEach((v, i) => { this.renderer.highlightNeuron(3, i, '#6f42c1'); this.renderer.showNeuronValue(3, i, v); });
    }

    // ─── HELPERS ──────────────────────────────────────────────────────────────

    _showBpView(show) {
        const bpView = document.getElementById('bpView');
        const defaultViz = document.getElementById('defaultViz');
        if (bpView) bpView.style.display = show ? 'flex' : 'none';
        if (defaultViz) defaultViz.style.display = show ? 'none' : 'block';
    }
}

// Global singleton
let bpController;

// Navigate to Activation Function step from within backprop
function goToActivationFunction() {
    const step3Btn = document.getElementById('step3Btn');
    if (step3Btn) step3Btn.click();
}
