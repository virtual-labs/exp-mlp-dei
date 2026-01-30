// ===================================
// GLOBAL STATE MANAGEMENT
// ===================================
const appState = {
    completedSteps: [],
    currentStep: 1,
    maxUnlockedStep: 1
};

// ===================================
// HYPERPARAMETER DATA FROM PDF
// ===================================
const trainingResults = {
    '50-8-SGD-0.01': { 
        accuracy: 0.7333, 
        macroF1: 0.7333, 
        precision: [1.000000, 0.600000, 0.600000], 
        recall: [1.000000, 0.600000, 0.600000],
        f1score: [1.000000, 0.600000, 0.600000],
        support: [10.000000, 10.000000, 10.000000]
    },
    '100-8-SGD-0.01': { 
        accuracy: 0.8000, 
        macroF1: 0.7802, 
        precision: [1.000, 1.000, 0.625], 
        recall: [1.0, 0.4, 1.0],
        f1score: [1.000000, 0.571429, 0.769231],
        support: [10.0, 10.0, 10.0]
    },
    '50-16-SGD-0.01': { 
        accuracy: 0.7667, 
        macroF1: 0.7277, 
        precision: [0.909091, 1.000000, 0.625000], 
        recall: [1.000000, 0.300000, 1.000000],
        f1score: [0.952381, 0.461538, 0.769231],
        support: [10.000000, 10.000000, 10.000000]
    },
    '100-16-SGD-0.01': { 
        accuracy: 0.9000, 
        macroF1: 0.8977, 
        precision: [1.000000, 1.000000, 0.769231], 
        recall: [1.0, 0.7, 1.0],
        f1score: [1.000000, 0.823529, 0.869565],
        support: [10.0, 10.0, 10.0]
    },
    '50-8-SGD-0.0005': { 
        accuracy: 0.5667, 
        macroF1: 0.4498, 
        precision: [0.588235, 0.000000, 0.538462], 
        recall: [1.000000, 0.000000, 0.700000],
        f1score: [0.740741, 0.000000, 0.608696],
        support: [10.000000, 10.000000, 10.000000]
    },
    '100-8-SGD-0.0005': { 
        accuracy: 0.6667, 
        macroF1: 0.6612, 
        precision: [0.818182, 0.555556, 0.600000], 
        recall: [0.900000, 0.500000, 0.600000],
        f1score: [0.857143, 0.526316, 0.600000],
        support: [10.000000, 10.000000, 10.000000]
    },
    '50-16-SGD-0.0005': { 
        accuracy: 0.5667, 
        macroF1: 0.4607, 
        precision: [0.777778, 0.476190, 0.000000], 
        recall: [0.700000, 1.000000, 0.000000],
        f1score: [0.736842, 0.645161, 0.000000],
        support: [10.000000, 10.000000, 10.000000]
    },
    '100-16-SGD-0.0005': { 
        accuracy: 0.3000, 
        macroF1: 0.1954, 
        precision: [0.333333, 0.296296, 0.000000], 
        recall: [0.1, 0.8, 0.0],
        f1score: [0.153846, 0.432432, 0.000000],
        support: [10.0, 10.0, 10.0]
    },
    '50-8-Adam-0.01': { 
        accuracy: 0.9333, 
        macroF1: 0.9333, 
        precision: [1.000000, 0.900000, 0.900000], 
        recall: [1.000000, 0.900000, 0.900000],
        f1score: [1.000000, 0.900000, 0.900000],
        support: [10.000000, 10.000000, 10.000000]
    },
    '100-8-Adam-0.01': { 
        accuracy: 0.9333, 
        macroF1: 0.9333, 
        precision: [1.000000, 0.900000, 0.900000], 
        recall: [1.000000, 0.900000, 0.900000],
        f1score: [1.000000, 0.900000, 0.900000],
        support: [10.000000, 10.000000, 10.000000]
    },
    '50-16-Adam-0.01': { 
        accuracy: 0.9333, 
        macroF1: 0.9333, 
        precision: [1.000000, 0.900000, 0.900000], 
        recall: [1.000000, 0.900000, 0.900000],
        f1score: [1.000000, 0.900000, 0.900000],
        support: [10.000000, 10.000000, 10.000000]
    },
    '100-16-Adam-0.01': { 
        accuracy: 0.9333, 
        macroF1: 0.9333, 
        precision: [1.000000, 0.900000, 0.900000], 
        recall: [1.000000, 0.900000, 0.900000],
        f1score: [1.000000, 0.900000, 0.900000],
        support: [10.000000, 10.000000, 10.000000]
    },
    '50-8-Adam-0.0005': { 
        accuracy: 0.8333, 
        macroF1: 0.8222, 
        precision: [1.000000, 1.000000, 0.666667], 
        recall: [1.000000, 0.500000, 1.000000],
        f1score: [1.000000, 0.666667, 0.800000],
        support: [10.000000, 10.000000, 10.000000]
    },
    '100-8-Adam-0.0005': { 
        accuracy: 0.9000, 
        macroF1: 0.8995, 
        precision: [1.000000, 0.888889, 0.833333], 
        recall: [0.9, 0.8, 1.0],
        f1score: [0.947368, 0.842105, 0.909091],
        support: [10.0, 10.0, 10.0]
    },
    '50-16-Adam-0.0005': { 
        accuracy: 0.7667, 
        macroF1: 0.7500, 
        precision: [1.000000, 0.800000, 0.625000], 
        recall: [0.900000, 0.400000, 1.000000],
        f1score: [0.947368, 0.533333, 0.769231],
        support: [10.000000, 10.000000, 10.000000]
    },
    '100-16-Adam-0.0005': { 
        accuracy: 0.8667, 
        macroF1: 0.8649, 
        precision: [1.000000, 0.875000, 0.769231], 
        recall: [0.900000, 0.700000, 1.000000],
        f1score: [0.947368, 0.777778, 0.869565],
        support: [10.000000, 10.000000, 10.000000]
    },
    '50-8-RMSprop-0.01': { 
        accuracy: 0.9333, 
        macroF1: 0.9333, 
        precision: [1.000000, 0.900000, 0.900000], 
        recall: [1.000000, 0.900000, 0.900000],
        f1score: [1.000000, 0.900000, 0.900000],
        support: [10.000000, 10.000000, 10.000000]
    },
    '100-8-RMSprop-0.01': { 
        accuracy: 0.9667, 
        macroF1: 0.9666, 
        precision: [1.000000, 0.909091, 1.000000], 
        recall: [1.000000, 1.000000, 0.900000],
        f1score: [1.000000, 0.952381, 0.947368],
        support: [10.000000, 10.000000, 10.000000]
    },
    '50-16-RMSprop-0.01': { 
        accuracy: 0.9333, 
        macroF1: 0.9332, 
        precision: [1.000000, 0.900000, 0.909091], 
        recall: [0.900000, 0.900000, 1.000000],
        f1score: [0.947368, 0.900000, 0.952381],
        support: [10.000000, 10.000000, 10.000000]
    },
    '100-16-RMSprop-0.01': { 
        accuracy: 0.9000, 
        macroF1: 0.8997, 
        precision: [1.000000, 0.818182, 0.888889], 
        recall: [1.0, 0.9, 0.8],
        f1score: [1.000000, 0.857143, 0.842105],
        support: [10.0, 10.0, 10.0]
    },
    '50-8-RMSprop-0.0005': { 
        accuracy: 0.8000, 
        macroF1: 0.7908, 
        precision: [1.000000, 0.833333, 0.666667], 
        recall: [0.9, 0.5, 1.0],
        f1score: [0.947368, 0.625000, 0.800000],
        support: [10.0, 10.0, 10.0]
    },
    '100-8-RMSprop-0.0005': { 
        accuracy: 0.9000, 
        macroF1: 0.8995, 
        precision: [1.000000, 0.888889, 0.833333], 
        recall: [0.9, 0.8, 1.0],
        f1score: [0.947368, 0.842105, 0.909091],
        support: [10.0, 10.0, 10.0]
    },
    '50-16-RMSprop-0.0005': { 
        accuracy: 0.8000, 
        macroF1: 0.7802, 
        precision: [1.000, 1.000, 0.625], 
        recall: [1.0, 0.4, 1.0],
        f1score: [1.000000, 0.571429, 0.769231],
        support: [10.0, 10.0, 10.0]
    },
    '100-16-RMSprop-0.0005': { 
        accuracy: 0.8667, 
        macroF1: 0.8611, 
        precision: [1.000000, 1.000000, 0.714286], 
        recall: [1.000000, 0.600000, 1.000000],
        f1score: [1.000000, 0.750000, 0.833333],
        support: [10.000000, 10.000000, 10.000000]
    }
};

// ===================================
// NEURAL NETWORK CLASS
// ===================================
class IrisNN {
    constructor() {
        this.layers = [4, 10, 8, 3];
        this.weights = [];
        this.biases = [];
        this.activations = [];
        this.z_values = [];
        this.gradients = { 
            weights: [], 
            biases: [], 
            activations: [] 
        };
        this.initializeWeights();
    }
    
    initializeWeights() {
        for (let i = 0; i < this.layers.length - 1; i++) {
            const inputSize = this.layers[i];
            const outputSize = this.layers[i + 1];
            const scale = Math.sqrt(2.0 / inputSize);
            
            const w = [];
            for (let j = 0; j < outputSize; j++) {
                const row = [];
                for (let k = 0; k < inputSize; k++) {
                    row.push((Math.random() - 0.5) * 2 * scale);
                }
                w.push(row);
            }
            this.weights.push(w);
            
            const b = new Array(outputSize).fill(0);
            this.biases.push(b);
        }
    }
    
    relu(x) { 
        return Math.max(0, x); 
    }
    
    reluDerivative(x) { 
        return x > 0 ? 1 : 0; 
    }
    
    softmax(arr) {
        const maxVal = Math.max(...arr);
        const exps = arr.map(x => Math.exp(x - maxVal));
        const sumExps = exps.reduce((a, b) => a + b, 0);
        return exps.map(x => x / sumExps);
    }
    
    forward(input) {
        this.activations = [input];
        this.z_values = [];
        
        for (let i = 0; i < this.weights.length; i++) {
            const z = [];
            for (let j = 0; j < this.weights[i].length; j++) {
                let sum = this.biases[i][j];
                for (let k = 0; k < this.weights[i][j].length; k++) {
                    sum += this.weights[i][j][k] * this.activations[i][k];
                }
                z.push(sum);
            }
            this.z_values.push(z);
            
            let a = (i < this.weights.length - 1) 
                ? z.map(val => this.relu(val))
                : this.softmax(z);
            this.activations.push(a);
        }
        
        return this.activations[this.activations.length - 1];
    }
    
    backward(target, learningRate) {
        const numLayers = this.weights.length;
        this.gradients.activations = [];
        
        const output = this.activations[numLayers];
        const outputGrad = output.map((o, i) => o - target[i]);
        this.gradients.activations[numLayers] = outputGrad;
        
        for (let l = numLayers - 1; l >= 0; l--) {
            this.gradients.weights[l] = [];
            this.gradients.biases[l] = [];
            
            const nextGrad = new Array(this.layers[l]).fill(0);
            
            for (let j = 0; j < this.weights[l].length; j++) {
                const grad_z = this.gradients.activations[l + 1][j];
                
                this.gradients.weights[l][j] = [];
                for (let k = 0; k < this.weights[l][j].length; k++) {
                    const grad_w = grad_z * this.activations[l][k];
                    this.gradients.weights[l][j][k] = grad_w;
                    nextGrad[k] += grad_z * this.weights[l][j][k];
                }
                
                this.gradients.biases[l][j] = grad_z;
            }
            
            if (l > 0) {
                this.gradients.activations[l] = nextGrad.map((g, i) => 
                    g * this.reluDerivative(this.z_values[l - 1][i])
                );
            }
        }
        
        for (let l = 0; l < this.weights.length; l++) {
            for (let j = 0; j < this.weights[l].length; j++) {
                for (let k = 0; k < this.weights[l][j].length; k++) {
                    this.weights[l][j][k] -= learningRate * this.gradients.weights[l][j][k];
                }
                this.biases[l][j] -= learningRate * this.gradients.biases[l][j];
            }
        }
    }
    
    getLoss(target) {
        const output = this.activations[this.activations.length - 1];
        let loss = 0;
        for (let i = 0; i < output.length; i++) {
            loss -= target[i] * Math.log(output[i] + 1e-10);
        }
        return loss;
    }
}

// ===================================
// GLOBAL VARIABLES
// ===================================
let network = new IrisNN();
let currentEpoch = 0;
let isTraining = false;
let trainingInterval = null;

// ===================================
// STEP MANAGEMENT FUNCTIONS
// ===================================
function unlockNextStep() {
    appState.maxUnlockedStep = Math.min(appState.maxUnlockedStep + 1, 6);
    updateStepButtons();
}

function setStepRunning(stepNum) {
    if (appState.completedSteps.includes(stepNum)) {
        return;
    }
    
    const stepBtn = document.querySelector(`.step-item[data-step="${stepNum}"]`);
    if (stepBtn) {
        stepBtn.classList.remove('idle', 'completed');
        stepBtn.classList.add('running');
    }
}

function completeStep(stepNum) {
    // 1. Update Sidebar Button
    const stepBtn = document.querySelector(`.step-item[data-step="${stepNum}"]`);
    if (stepBtn) {
        stepBtn.classList.remove('idle', 'running');
        stepBtn.classList.add('completed');
    }

    // 2. NEW: Update Main Content Block (The Notebook Cell)
    const stepContent = document.getElementById(`step-${stepNum}`);
    if (stepContent) {
        stepContent.classList.add('completed');
    }
    
    // 3. Update State
    if (!appState.completedSteps.includes(stepNum)) {
        appState.completedSteps.push(stepNum);
        unlockNextStep();
    }
}

function updateStepButtons() {
    const stepButtons = document.querySelectorAll('.step-item');
    stepButtons.forEach(btn => {
        const stepNum = parseInt(btn.dataset.step);
        
        if (appState.completedSteps.includes(stepNum)) {
            // CHANGE: Apply light green style directly to completed steps
            btn.style.backgroundColor = '#d1e7dd'; 
            btn.style.borderColor = '#badbcc'; 
            btn.style.color = '#0f5132'; 
            
            if (!btn.classList.contains('completed')) {
                btn.classList.add('completed', 'unlocked');
            }
            return;
        }
        
        // Reset styles for non-completed steps if needed
        btn.style.backgroundColor = ''; 
        btn.style.borderColor = ''; 
        btn.style.color = '';
        
        if (!btn.classList.contains('running') && !btn.classList.contains('completed')) {
            btn.classList.remove('active', 'unlocked', 'idle');
            
            if (stepNum === appState.currentStep) {
                btn.classList.add('active', 'unlocked', 'idle');
            } else if (stepNum <= appState.maxUnlockedStep) {
                btn.classList.add('unlocked', 'idle');
            } else {
                btn.classList.add('idle');
            }
        }
    });
}

// ===================================
// STEP NAVIGATION
// ===================================
const stepButtons = document.querySelectorAll('.step-item');
const stepContents = document.querySelectorAll('.step-content');

stepButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const stepNum = parseInt(btn.dataset.step);
        
        if (!btn.classList.contains('unlocked')) {
            alert('Please complete the current step first!');
            return;
        }
        
        const isCompleted = appState.completedSteps.includes(stepNum);
        
        appState.currentStep = stepNum;
        
        stepContents.forEach(c => c.classList.remove('active'));
        document.getElementById(`step-${stepNum}`).classList.add('active');
        
        if (!isCompleted) {
            updateStepButtons();
        }
        
        if (stepNum === 4) {
            displayArchitecture();
        } else if (stepNum === 5) {
            initializeCanvas();
        } else if (stepNum === 6) {
            displayResults();
        }
    });
});

// ===================================
// UPDATE CODE DISPLAY
// ===================================
function updateCodeDisplay() {
    const epochs = document.getElementById('epochs').value;
    const batchSize = document.getElementById('batchSize').value;
    const optimizer = document.getElementById('optimizer').value;
    const learningRate = document.getElementById('learningRate').value;
    
    const code = `history = model.fit(
    X_train, y_train,
    epochs=<span class="highlight">${epochs}</span>,
    batch_size=<span class="highlight">${batchSize}</span>,
    validation_split=<span class="number">0.1</span>,
    verbose=<span class="number">0</span>
)

model.compile(
    optimizer=tf.keras.optimizers.<span class="highlight">${optimizer}</span>(
        learning_rate=<span class="highlight">${learningRate}</span>
    ),
    loss=<span class="string">"categorical_crossentropy"</span>,
    metrics=[<span class="string">"accuracy"</span>]
)

<span class="builtin">print</span>(<span class="string">"Training Completed!"</span>)`;
    
    const codeElement = document.getElementById('trainingCode');
    if (codeElement) {
        codeElement.innerHTML = code;
    }
}

// ===================================
// PARAMETER SELECTS UPDATES
// ===================================
const paramSelects = document.querySelectorAll('.param-select');
paramSelects.forEach(select => {
    select.addEventListener('change', updateCodeDisplay);
});

// ===================================
// STEP 4: DISPLAY ARCHITECTURE
// ===================================
function displayArchitecture() {
    const arch = document.getElementById('architecture');
    if (arch) {
        arch.innerHTML = `
            <div style="font-size: 1.2em; line-height: 2.5;">
                <strong>Input Layer:</strong> 4 neurons<br>
                ↓<br>
                <strong>Dense(10, activation='relu')</strong><br>
                ↓<br>
                <strong>Dense(8, activation='relu')</strong><br>
                ↓<br>
                <strong>Output Layer:</strong> 3 neurons (Softmax)<br><br>
                <em>Architecture: 4 → 10 → 8 → 3</em>
            </div>
        `;
    }
}

// ===================================
// STEP 5: CANVAS DRAWING
// ===================================
const canvas = document.getElementById('networkCanvas');
const ctx = canvas ? canvas.getContext('2d') : null;

function initializeCanvas() {
    if (canvas && ctx) {
        resizeCanvas();
    }
}

function resizeCanvas() {
    if (!canvas) return;
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    drawNetwork();
}

if (canvas) {
    window.addEventListener('resize', resizeCanvas);
}

function drawNetwork() {
    if (!ctx || !canvas) return;
    
    if (canvas.offsetWidth === 0 || canvas.offsetHeight === 0) {
        return;
    }
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const layerSpacing = canvas.width / (network.layers.length + 1);
    const nodePositions = [];
    
    for (let l = 0; l < network.layers.length; l++) {
        const x = layerSpacing * (l + 1);
        const spacing = canvas.height / (network.layers[l] + 1);
        const positions = [];
        
        for (let n = 0; n < network.layers[l]; n++) {
            positions.push({ x, y: spacing * (n + 1) });
        }
        nodePositions.push(positions);
    }
    
    for (let l = 0; l < network.weights.length; l++) {
        for (let j = 0; j < network.weights[l].length; j++) {
            for (let k = 0; k < network.weights[l][j].length; k++) {
                const from = nodePositions[l][k];
                const to = nodePositions[l + 1][j];
                const weight = network.weights[l][j][k];
                const gradient = Math.abs(network.gradients.weights[l]?.[j]?.[k] || 0);
                
                const thickness = Math.max(0.5, Math.min(Math.abs(weight) * 2, 4));
                
                let color;
                if (isTraining && gradient > 0.01) {
                    const pulse = Math.sin(Date.now() / 200) * 0.3 + 0.7;
                    color = `rgba(243, 156, 18, ${pulse})`;
                } else {
                    color = weight > 0 ? '#27ae60' : '#e74c3c';
                }
                
                ctx.beginPath();
                ctx.moveTo(from.x, from.y);
                ctx.lineTo(to.x, to.y);
                ctx.strokeStyle = color;
                ctx.lineWidth = thickness;
                ctx.globalAlpha = 0.6;
                ctx.stroke();
                ctx.globalAlpha = 1;
            }
        }
    }
    
    for (let l = 0; l < nodePositions.length; l++) {
        for (let n = 0; n < nodePositions[l].length; n++) {
            const pos = nodePositions[l][n];
            const activation = network.activations[l]?.[n] || 0;
            const intensity = Math.min(Math.abs(activation), 1);
            
            let radius = 18;
            if (isTraining && activation > 0.1) {
                const pulse = Math.sin(Date.now() / 150 + n * 0.5) * 2 + 18;
                radius = pulse;
            }
            
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
            ctx.fillStyle = activation > 0 
                ? `rgba(46, 204, 113, ${0.3 + intensity * 0.5})` 
                : `rgba(231, 76, 60, ${0.3 + intensity * 0.5})`;
            ctx.fill();
            ctx.strokeStyle = '#2c3e50';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            ctx.fillStyle = '#2c3e50';
            ctx.font = 'bold 11px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(activation.toFixed(2), pos.x, pos.y);
            
            if (n === 0) {
                ctx.fillStyle = '#555';
                ctx.font = 'bold 12px Arial';
                const labels = ['Input', 'Hidden 1', 'Hidden 2', 'Output'];
                ctx.fillText(labels[l], pos.x, 20);
            }
        }
    }
    
    // CHANGE THIS BLOCK:
    // Old: if (isTraining) { drawDataFlow(nodePositions); }
    
    // New: Run if training OR if there are leftover particles
    // Change: Continue drawing if training is active OR if particles remain
    if (isTraining || flowParticles.length > 0) {
        drawDataFlow(nodePositions);
    }
}


let flowParticles = [];

function drawDataFlow(nodePositions) {
    // Change: Check isTraining to stop spawning new dots when done.
    // Change: '0.4' makes dots appear more frequently (was 0.7).
    // Change: '50' allows more dots on screen at once (was 20).
    if (isTraining && Math.random() > 0.4 && flowParticles.length < 50) {
        flowParticles.push({
            // ...
            layer: 0,
            progress: 0,
            fromNode: Math.floor(Math.random() * network.layers[0]),
            toNode: Math.floor(Math.random() * network.layers[1])
        });
    }
    
    flowParticles = flowParticles.filter(particle => {
        particle.progress += 0.05;
        
        if (particle.progress >= 1) {
            particle.layer++;
            particle.progress = 0;
            particle.fromNode = particle.toNode;
            if (particle.layer < network.weights.length) {
                particle.toNode = Math.floor(Math.random() * network.layers[particle.layer + 1]);
            } else {
                return false; // Remove when they reach the end
            }
        }
        
        if (particle.layer < network.weights.length) {
            // ... existing drawing code (no changes needed here) ...
            const from = nodePositions[particle.layer][particle.fromNode];
            const to = nodePositions[particle.layer + 1][particle.toNode];
            
            const x = from.x + (to.x - from.x) * particle.progress;
            const y = from.y + (to.y - from.y) * particle.progress;
            
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fillStyle = '#3498db';
            ctx.fill();
            
            ctx.beginPath();
            ctx.arc(x, y, 8, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(52, 152, 219, 0.3)';
            ctx.fill();
        }
        
        return particle.layer < network.weights.length;
    });
}
// ===================================
// STEP 5: TRAINING
// ===================================
const trainBtn = document.getElementById('trainBtn');
if (trainBtn) {
    trainBtn.addEventListener('click', () => {
        if (isTraining) {
            stopTraining();
        } else {
            startTraining();
        }
    });
}

function startTraining() {
    if (isTraining) return;
    
    isTraining = true;
    currentEpoch = 0;
    
    const epochs = parseInt(document.getElementById('epochs').value);
    const batchSize = parseInt(document.getElementById('batchSize').value);
    const optimizer = document.getElementById('optimizer').value;
    const lr = parseFloat(document.getElementById('learningRate').value);
    
    const key = `${epochs}-${batchSize}-${optimizer}-${lr}`;
    const results = trainingResults[key];
    
    if (!results) {
        alert('Configuration not found in training data!');
        isTraining = false;
        return;
    }
    
    setStepRunning(5);
    
    trainBtn.textContent = 'Training...';
    trainBtn.classList.add('running');
    trainBtn.disabled = true;
    
    const codeBlock = document.querySelector('#step-5 .code-block');
    const outputBox = document.querySelector('#step-5 .output-box');
    if (codeBlock) codeBlock.classList.add('running');
    if (outputBox) outputBox.classList.add('running');
    
    const outputContent = document.getElementById('output-5');
    if (outputContent) {
        outputContent.style.display = 'block';
    }
    
    setTimeout(() => {
        if (canvas) {
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
            drawNetwork();
        }
    }, 50);
    
    document.getElementById('trainingStatus').textContent = 'Training in progress...';
    
    const epochDuration = 3000 / epochs;
    trainingInterval = setInterval(() => {
        currentEpoch++;
        
        const progress = currentEpoch / epochs;
        const startLoss = 1.2;
        const endLoss = -(results.accuracy - 1) * 0.5;
        const loss = startLoss - (startLoss - endLoss) * progress;
        
        const startAcc = 0.33;
        const acc = startAcc + (results.accuracy - startAcc) * progress;
        
        const inputs = [0.1, 0.5, 0.2, 0.05];
        const target = [1, 0, 0];
        
        network.forward(inputs);
        network.backward(target, lr);
        
        document.getElementById('currentEpoch').textContent = currentEpoch;
        document.getElementById('lossValue').textContent = loss.toFixed(6);
        document.getElementById('accuracy').textContent = acc.toFixed(4);
        
        drawNetwork();
        
        if (currentEpoch >= epochs) {
            stopTraining();
            completeTraining(results);
        }
    }, epochDuration);
}

function stopTraining() {
    if (trainingInterval) {
        clearInterval(trainingInterval);
        trainingInterval = null;
    }
    isTraining = false;
    
    // Change: Removed 'flowParticles = []' so dots don't vanish.
    
    // New: Helper function to keep animating until all dots are gone
    function finishAnimation() {
        if (flowParticles.length > 0) {
            drawNetwork(); // Updates position of remaining dots
            requestAnimationFrame(finishAnimation); // Keep loop going
        }
    }
    
    // Start the finish loop
    requestAnimationFrame(finishAnimation);
}

function completeTraining(results) {
    // 1. Existing logic for the top button
    trainBtn.textContent = 'Next →';
    trainBtn.classList.remove('running');
    trainBtn.classList.add('completed');
    trainBtn.disabled = false;
    
    const codeBlock = document.querySelector('#step-5 .code-block');
    const outputBox = document.querySelector('#step-5 .output-box');
    if (codeBlock) {
        codeBlock.classList.remove('running');
        codeBlock.classList.add('completed');
    }
    if (outputBox) {
        outputBox.classList.remove('running');
        outputBox.classList.add('completed');
    }
    
    document.getElementById('accuracy').textContent = results.accuracy.toFixed(4);
    document.getElementById('trainingStatus').innerHTML = `✓ Training Completed Successfully!<br><br>
        <strong>Final Results:</strong><br>
        Accuracy: ${results.accuracy.toFixed(4)}<br>
        Macro F1-Score: ${results.macroF1.toFixed(4)}`;
    
    window.currentResults = results;
    
    completeStep(5);
    
    // --- NEW LOGIC STARTS HERE ---
    
    // Define the navigation action
    const goToNextStep = function() {
        const nextStepBtn = document.querySelector('.step-item[data-step="6"]');
        if (nextStepBtn) {
            appState.currentStep = 6;
            stepContents.forEach(c => c.classList.remove('active'));
            
            const step6 = document.getElementById('step-6');
            step6.classList.add('active');
            
            // Scroll logic
            step6.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            displayResults();
            updateStepButtons();
        }
    };

    // 1. Attach to Top Button
    trainBtn.onclick = goToNextStep;

    // 2. Attach to Bottom Button (The Fix)
    const bottomNav = document.getElementById('step5-bottom-nav');
    const bottomBtn = document.getElementById('btnNextStep5');
    
    if (bottomNav && bottomBtn) {
        bottomNav.style.display = 'block'; // Reveal the button
        bottomBtn.onclick = goToNextStep;  // Attach same logic
        
        // Auto-scroll slightly to make sure the user sees the new button
        bottomNav.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

// ===================================
// STEP 6: DISPLAY RESULTS
// ===================================
function displayResults() {
    const results = window.currentResults || trainingResults['50-8-SGD-0.01'];
    
    updateResultsTable(results);
    drawConfusionMatrix(results);
}

function updateResultsTable(results) {
    const tbody = document.getElementById('resultsTableBody');
    if (!tbody) return;
    
    const species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'];
    
    let html = '';
    
    for (let i = 0; i < 3; i++) {
        html += `<tr>
            <td>${species[i]}</td>
            <td>${results.precision[i].toFixed(6)}</td>
            <td>${results.recall[i].toFixed(6)}</td>
            <td>${results.f1score[i].toFixed(6)}</td>
            <td>${results.support[i].toFixed(6)}</td>
        </tr>`;
    }
    
    html += `<tr class="summary-row">
        <td>accuracy</td>
        <td>${results.accuracy.toFixed(6)}</td>
        <td>${results.accuracy.toFixed(6)}</td>
        <td>${results.accuracy.toFixed(6)}</td>
        <td>${results.support.reduce((a, b) => a + b, 0).toFixed(6)}</td>
    </tr>`;
    
    const avgPrecision = results.precision.reduce((a, b) => a + b, 0) / 3;
    const avgRecall = results.recall.reduce((a, b) => a + b, 0) / 3;
    const avgF1 = results.f1score.reduce((a, b) => a + b, 0) / 3;
    const totalSupport = results.support.reduce((a, b) => a + b, 0);
    
    html += `<tr>
        <td>macro avg</td>
        <td>${avgPrecision.toFixed(6)}</td>
        <td>${avgRecall.toFixed(6)}</td>
        <td>${avgF1.toFixed(6)}</td>
        <td>${totalSupport.toFixed(6)}</td>
    </tr>`;
    
    html += `<tr>
        <td>weighted avg</td>
        <td>${avgPrecision.toFixed(6)}</td>
        <td>${avgRecall.toFixed(6)}</td>
        <td>${avgF1.toFixed(6)}</td>
        <td>${totalSupport.toFixed(6)}</td>
    </tr>`;
    
    tbody.innerHTML = html;
}

function drawConfusionMatrix(results) {
    const container = document.getElementById('confusionMatrix');
    if (!container) return;
    
    const confMatrix = [
        [Math.round(results.recall[0] * 10), Math.round((1 - results.recall[0]) * 10 * 0.5), Math.round((1 - results.recall[0]) * 10 * 0.5)],
        [Math.round((1 - results.recall[1]) * 10 * 0.5), Math.round(results.recall[1] * 10), Math.round((1 - results.recall[1]) * 10 * 0.5)],
        [Math.round((1 - results.recall[2]) * 10 * 0.5), Math.round((1 - results.recall[2]) * 10 * 0.5), Math.round(results.recall[2] * 10)]
    ];
    
    let html = '<h3 style="margin-bottom: 15px; color: #2c3e50;">Confusion Matrix</h3>';
    html += '<table style="width: 100%; max-width: 500px; margin: 0 auto; border-collapse: collapse;">';
    html += '<tr><th style="border: 2px solid #2c3e50; padding: 10px; background: #ecf0f1;"></th>';
    html += '<th style="border: 2px solid #2c3e50; padding: 10px; background: #ecf0f1;">Setosa</th>';
    html += '<th style="border: 2px solid #2c3e50; padding: 10px; background: #ecf0f1;">Versicolor</th>';
    html += '<th style="border: 2px solid #2c3e50; padding: 10px; background: #ecf0f1;">Virginica</th></tr>';
    
    const labels = ['Setosa', 'Versicolor', 'Virginica'];
    for (let i = 0; i < 3; i++) {
        html += `<tr><td style="border: 2px solid #2c3e50; padding: 10px; background: #ecf0f1; font-weight: 700;">${labels[i]}</td>`;
        for (let j = 0; j < 3; j++) {
            const val = confMatrix[i][j];
            const bg = i === j ? '#d5f4e6' : (val > 0 ? '#fadbd8' : '#ffffff');
            html += `<td style="border: 2px solid #2c3e50; padding: 15px; text-align: center; background: ${bg}; font-weight: 700; font-size: 1.2em;">${val}</td>`;
        }
        html += '</tr>';
    }
    html += '</table>';
    
    html += `<div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border: 2px solid #dee2e6; border-radius: 4px;">
        <h4 style="color: #2c3e50; margin-bottom: 10px;">Performance Metrics:</h4>
        <p style="font-size: 1.1em; line-height: 1.8;">
            <strong>Accuracy:</strong> ${results.accuracy.toFixed(4)}<br>
            <strong>Macro F1-Score:</strong> ${results.macroF1.toFixed(4)}<br><br>
            <strong>Per-Class Precision:</strong><br>
            Setosa: ${results.precision[0].toFixed(3)}<br>
            Versicolor: ${results.precision[1].toFixed(3)}<br>
            Virginica: ${results.precision[2].toFixed(3)}
        </p>
    </div>`;
    
    container.innerHTML = html;
}

// ===================================
// RUN BUTTONS FOR EACH STEP
// ===================================
function setupRunButtons() {
    const runButtons = document.querySelectorAll('.run-btn');
    runButtons.forEach((btn) => {
        if (btn.id === 'trainBtn') return;
        
        btn.addEventListener('click', function() {
            const stepContent = this.closest('.step-content');
            const stepId = stepContent.id;
            const stepNum = parseInt(stepId.split('-')[1]);
            
            if (appState.completedSteps.includes(stepNum)) {
                if (stepNum < 6) {
                    appState.currentStep = stepNum + 1;
                    stepContents.forEach(c => c.classList.remove('active'));
                    document.getElementById(`step-${stepNum + 1}`).classList.add('active');

                    // --- NEW CODE START ---
        const nextStepEl = document.getElementById(`step-${stepNum + 1}`);
        nextStepEl.classList.add('active');
        
        // This command forces the browser to scroll
        nextStepEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
        // --- NEW CODE END ---
                    
                    if (stepNum + 1 === 4) {
                        displayArchitecture();
                    } else if (stepNum + 1 === 5) {
                        initializeCanvas();
                    } else if (stepNum + 1 === 6) {
                        displayResults();
                    }
                    
                    updateStepButtons();
                }
                return;
            }
            
            setStepRunning(stepNum);
            
            this.textContent = 'Running...';
            this.classList.add('running');
            
            const codeBlock = stepContent.querySelector('.code-block');
            const outputBox = stepContent.querySelector('.output-box');
            
            if (codeBlock) codeBlock.classList.add('running');
            if (outputBox) outputBox.classList.add('running');
            
            setTimeout(() => {
                this.textContent = 'Next →';
                this.classList.remove('running');
                this.classList.add('completed');
                
                if (codeBlock) {
                    codeBlock.classList.remove('running');
                    codeBlock.classList.add('completed');
                }
                if (outputBox) {
                    outputBox.classList.remove('running');
                    outputBox.classList.add('completed');
                }
                
                const outputContent = document.getElementById(`output-${stepNum}`);
                if (outputContent) {
                    outputContent.style.display = 'block';
                }
                
                if (stepNum === 4) {
                    displayArchitecture();
                } else if (stepNum === 6) {
                    displayResults();
                }
                
                completeStep(stepNum);
                
                this.onclick = function() {
                    if (stepNum < 6) {
                        appState.currentStep = stepNum + 1;
                        stepContents.forEach(c => c.classList.remove('active'));
                        document.getElementById(`step-${stepNum + 1}`).classList.add('active');

                        // ADD THIS LINE:
                        nextStepEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        
                        if (stepNum + 1 === 4) {
                            displayArchitecture();
                        } else if (stepNum + 1 === 5) {
                            initializeCanvas();
                        } else if (stepNum + 1 === 6) {
                            displayResults();
                        }
                        
                        updateStepButtons();
                    }
                };
            }, 1000);
        });
    });
}

// ===================================
// INITIALIZATION
// ===================================
document.addEventListener('DOMContentLoaded', () => {
    displayArchitecture();
    if (canvas) {
        initializeCanvas();
    }
    setupRunButtons();
    updateStepButtons();
});


// ===================================
// LAYOUT NAVIGATION & SCROLL LOGIC
// ===================================

/**
 * Handles sidebar clicks to scroll to specific notebook sections
 */
function initializeScrolling() {
    const sidebarItems = document.querySelectorAll('.step-item');
    
    sidebarItems.forEach(item => {
        item.addEventListener('click', () => {
            const stepId = `step-${item.dataset.step}`;
            const targetElement = document.getElementById(stepId);
            
            if (targetElement) {
                targetElement.scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'start' 
                });
                
                // Update active state in state management
                appState.currentStep = parseInt(item.dataset.step);
                updateStepButtons();
            }
        });
    });
}

/**
 * Optional: Intersection Observer to update sidebar active state while scrolling
 */
const scrollObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const stepNum = parseInt(entry.target.id.split('-')[1]);
            appState.currentStep = stepNum;
            updateStepButtons();
        }
    });
}, { threshold: 0.5 });

// Start observing all steps
document.querySelectorAll('.step-content').forEach(section => {
    scrollObserver.observe(section);
});

// Re-initialize UI for scrolling layout
document.addEventListener('DOMContentLoaded', () => {
    initializeScrolling();
    
    // Ensure Step 5 canvas is visible when scrolled to
    const step5 = document.getElementById('step-5');
    if(step5) {
        step5.addEventListener('mouseenter', () => {
            if (canvas && isTraining) drawNetwork();
        });
    }
});

// ============================================================
// UI RESTRUCTURING & ACTION LOGIC
// ============================================================

/**
 * Transforms the existing DOM to match the reference image layout.
 * Moves Run buttons to a new Header bar and creates a vertical stack.
 */
function applyReferenceLayout() {
    const steps = document.querySelectorAll('.step-content');
    
    steps.forEach(step => {
        // 1. Get Step info
        const stepId = step.id; // e.g., "step-1"
        const stepNum = stepId.split('-')[1];
        
        // 2. Find Title from Sidebar
        const sidebarItem = document.querySelector(`.step-item[data-step="${stepNum}"]`);
        const titleText = sidebarItem.querySelector('.step-title').textContent.trim();
        
        // 3. Create Header Container
        const header = document.createElement('div');
        header.className = 'notebook-cell-header';
        
        // 4. Create Title
        const titleDiv = document.createElement('div');
        titleDiv.className = 'notebook-cell-title';
        titleDiv.textContent = `Step ${stepNum}: ${titleText}`;
        
        // 5. Find and Move the Run Button
        const runBtn = step.querySelector('.run-btn');
        if (runBtn) {
            // Update button text to match reference if needed
            if (runBtn.textContent.trim() === 'Run') {
                runBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" style="margin-right:5px"><path d="M8 5v14l11-7z"/></svg> Run';
            }
            header.appendChild(titleDiv);
            header.appendChild(runBtn); // Moves button from code-panel to header
        }
        
        // 6. Insert Header at top of step
        step.insertBefore(header, step.firstChild);
        
        // 7. Add "OUTPUT:" label to output box if missing
        const outputBox = step.querySelector('.output-box');
        if (outputBox && !outputBox.querySelector('.output-label-styled')) {
            const label = document.createElement('div');
            label.className = 'output-label-styled';
            label.textContent = 'OUTPUT:';
            outputBox.insertBefore(label, outputBox.firstChild);
        }
    });

    // 8. Add Scroll Navigation
    // Ensure clicking sidebar scrolls to the step instead of just toggling visibility
    const sidebarLinks = document.querySelectorAll('.step-item');
    sidebarLinks.forEach(link => {
        // Clone to remove old event listeners (optional, but safer to prevent conflicts)
        const newLink = link.cloneNode(true);
        link.parentNode.replaceChild(newLink, link);
        
        newLink.addEventListener('click', (e) => {
            const sNum = parseInt(newLink.dataset.step);
            
            // Allow navigation if step is unlocked or completed
            if (appState.completedSteps.includes(sNum - 1) || sNum === 1 || appState.completedSteps.includes(sNum)) {
                document.getElementById(`step-${sNum}`).scrollIntoView({ behavior: 'smooth', block: 'center' });
                // Update active state visual only
                document.querySelectorAll('.step-item').forEach(i => i.classList.remove('active'));
                newLink.classList.add('active');
            }
        });
    });
}

/**
 * Injects "Run All" and "Reset" buttons into the sidebar
 */
function injectSidebarActions() {
    const sidebar = document.querySelector('.step-sidebar');
    if (!sidebar) return;

    const footer = document.createElement('div');
    footer.className = 'sidebar-footer';
    
    footer.innerHTML = `
        <button id="btnRunAll" class="action-btn btn-run-all">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
            Run All
        </button>
        <button id="btnReset" class="action-btn btn-reset">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2.5 2v6h6M21.5 22v-6h-6"/><path d="M22 11.5A10 10 0 0 0 3.2 7.2M2 12.5a10 10 0 0 0 18.8 4.3"/></svg>
            Reset
        </button>
    `;
    
    sidebar.appendChild(footer);
    
    // Bind Events
    document.getElementById('btnRunAll').addEventListener('click', runAllSteps);
    document.getElementById('btnReset').addEventListener('click', () => window.location.reload());
}

/**
 * Logic to execute all steps sequentially
 */
async function runAllSteps() {
    const runBtn = document.getElementById('btnRunAll');
    if(runBtn.disabled) return;
    
    runBtn.disabled = true;
    runBtn.innerHTML = 'Running...';
    
    const totalSteps = 6;
    
    for (let i = 1; i <= totalSteps; i++) {
        // Scroll to current step
        const stepElem = document.getElementById(`step-${i}`);
        stepElem.scrollIntoView({ behavior: 'smooth', block: 'center' });
        
        // If already completed, skip
        if (appState.completedSteps.includes(i)) continue;
        
        // Find the run button for this step
        // Note: We moved it to .notebook-cell-header in applyReferenceLayout
        const cellBtn = stepElem.querySelector('.run-btn');
        
        if (cellBtn) {
            // Trigger click and wait
            cellBtn.click();
            
            // Wait for completion (heuristic: wait until button says "Next" or has "completed" class)
            await new Promise(resolve => {
                const checkInterval = setInterval(() => {
                    if (cellBtn.classList.contains('completed') || appState.completedSteps.includes(i)) {
                        clearInterval(checkInterval);
                        setTimeout(resolve, 500); // Small delay between steps
                    }
                }, 200);
            });
        }
    }
    
    runBtn.innerHTML = '✓ Done';
    setTimeout(() => {
        runBtn.disabled = false;
        runBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg> Run All`;
    }, 2000);
}

// Initialize on Load
document.addEventListener('DOMContentLoaded', () => {
    applyReferenceLayout();
    injectSidebarActions();
});