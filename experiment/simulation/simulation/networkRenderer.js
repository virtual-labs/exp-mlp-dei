/**
 * networkRenderer.js
 * Renders the neural network using SVG
 */

class NetworkRenderer {
    constructor(svgId, mlpData) {
        this.svg = document.getElementById(svgId);
        this.data = mlpData;
        this.layers = mlpData.architecture.layers;
        this.nodePositions = [];
        this.connections = [];

        this.config = {
            width: 1000,
            height: 600,
            nodeRadius: 20,
            layerSpacing: 250,
            nodeSpacing: 60,
            colors: {
                neuronInactive: '#e9ecef',
                neuronActive: '#0d6efd',
                neuronHighlight: '#ffc107',
                connectionInactive: '#dee2e6',
                connectionActive: '#0d6efd',
                connectionGradient: '#f39c12',
                positiveGradient: '#27ae60',
                negativeGradient: '#e74c3c'
            }
        };

        this.initialize();
    }

    initialize() {
        this.svg.setAttribute('viewBox', `0 0 ${this.config.width} ${this.config.height}`);
        this.calculatePositions();
        this.drawNetwork();
    }

    calculatePositions() {
        this.nodePositions = [];
        const totalLayers = this.layers.length;
        const startX = 100;

        for (let l = 0; l < totalLayers; l++) {
            const layerSize = this.layers[l];
            const x = startX + l * this.config.layerSpacing;
            // Increased spacing to prevent overlap with labels
            const nodeSpacing = l === 1 ? 50 : this.config.nodeSpacing; // Tighter spacing for layer 1
            const totalHeight = (layerSize - 1) * nodeSpacing;
            const startY = (this.config.height - totalHeight) / 2;

            const positions = [];
            for (let n = 0; n < layerSize; n++) {
                positions.push({
                    x: x,
                    y: startY + n * nodeSpacing,
                    layer: l,
                    index: n
                });
            }
            this.nodePositions.push(positions);
        }
    }

    drawNetwork() {
        // Clear SVG
        this.svg.innerHTML = '';

        // Create groups for layering
        const connectionGroup = this.createSVGElement('g', { id: 'connections' });
        const nodeGroup = this.createSVGElement('g', { id: 'nodes' });
        const labelGroup = this.createSVGElement('g', { id: 'labels' });

        // Draw connections
        this.drawConnections(connectionGroup);

        // Draw nodes
        this.drawNodes(nodeGroup);

        // Draw labels
        this.drawLabels(labelGroup);

        // Append to SVG
        this.svg.appendChild(connectionGroup);
        this.svg.appendChild(nodeGroup);
        this.svg.appendChild(labelGroup);
    }

    drawConnections(group) {
        this.connections = [];

        for (let l = 0; l < this.layers.length - 1; l++) {
            const fromLayer = this.nodePositions[l];
            const toLayer = this.nodePositions[l + 1];

            for (let i = 0; i < fromLayer.length; i++) {
                for (let j = 0; j < toLayer.length; j++) {
                    const line = this.createSVGElement('line', {
                        x1: fromLayer[i].x,
                        y1: fromLayer[i].y,
                        x2: toLayer[j].x,
                        y2: toLayer[j].y,
                        stroke: this.config.colors.connectionInactive,
                        'stroke-width': '1.5',
                        'stroke-opacity': '0.3',
                        class: `connection layer-${l}-${l + 1} from-${i} to-${j}`
                    });

                    this.connections.push({
                        element: line,
                        fromLayer: l,
                        toLayer: l + 1,
                        fromNode: i,
                        toNode: j
                    });

                    group.appendChild(line);
                }
            }
        }
    }

    drawNodes(group) {
        const layerLabels = ['Input', 'Hidden 1', 'Hidden 2', 'Output'];

        for (let l = 0; l < this.nodePositions.length; l++) {
            for (let n = 0; n < this.nodePositions[l].length; n++) {
                const pos = this.nodePositions[l][n];

                // Neuron circle
                const circle = this.createSVGElement('circle', {
                    cx: pos.x,
                    cy: pos.y,
                    r: this.config.nodeRadius,
                    fill: this.config.colors.neuronInactive,
                    stroke: '#adb5bd',
                    'stroke-width': '2',
                    class: `neuron layer-${l} node-${n}`,
                    'data-layer': l,
                    'data-node': n
                });

                group.appendChild(circle);

                // Activation value text (initially hidden)
                const text = this.createSVGElement('text', {
                    x: pos.x,
                    y: pos.y,
                    'text-anchor': 'middle',
                    'dominant-baseline': 'middle',
                    fill: '#212529',
                    'font-size': '11',
                    'font-weight': 'bold',
                    class: `neuron-value layer-${l} node-${n}`,
                    style: 'display: none;'
                });
                text.textContent = '0.00';

                group.appendChild(text);
            }
        }
    }

    drawLabels(group) {
        const layerLabels = ['Input Layer', 'Hidden Layer 1', 'Hidden Layer 2', 'Output Layer'];

        for (let l = 0; l < this.nodePositions.length; l++) {
            const pos = this.nodePositions[l][0];
            const text = this.createSVGElement('text', {
                x: pos.x,
                y: 30,
                'text-anchor': 'middle',
                fill: '#495057',
                'font-size': '14',
                'font-weight': '600'
            });
            text.textContent = layerLabels[l];
            group.appendChild(text);
        }
    }

    createSVGElement(tag, attrs) {
        const elem = document.createElementNS('http://www.w3.org/2000/svg', tag);
        for (let key in attrs) {
            elem.setAttribute(key, attrs[key]);
        }
        return elem;
    }

    // Animation methods
    highlightNeuron(layer, node, color = null) {
        const neuron = this.svg.querySelector(`.neuron.layer-${layer}.node-${node}`);
        if (neuron) {
            neuron.setAttribute('fill', color || this.config.colors.neuronActive);
            neuron.setAttribute('stroke', color || this.config.colors.neuronActive);
            neuron.setAttribute('stroke-width', '3');
        }
    }

    showNeuronValue(layer, node, value) {
        const text = this.svg.querySelector(`.neuron-value.layer-${layer}.node-${node}`);
        if (text) {
            text.textContent = value.toFixed(2);
            text.style.display = 'block';
        }
    }

    highlightConnection(fromLayer, fromNode, toLayer, toNode, color = null) {
        const conn = this.svg.querySelector(
            `.connection.layer-${fromLayer}-${toLayer}.from-${fromNode}.to-${toNode}`
        );
        if (conn) {
            conn.setAttribute('stroke', color || this.config.colors.connectionActive);
            conn.setAttribute('stroke-width', '3');
            conn.setAttribute('stroke-opacity', '0.8');
        }
    }

    animateConnectionFlow(fromLayer, fromNode, toLayer, toNode, duration = 500) {
        const conn = this.svg.querySelector(
            `.connection.layer-${fromLayer}-${toLayer}.from-${fromNode}.to-${toNode}`
        );
        if (conn) {
            const length = conn.getTotalLength();
            conn.style.strokeDasharray = length;
            conn.style.strokeDashoffset = length;
            conn.style.animation = `flowForward ${duration}ms ease-out forwards`;
        }
    }

    resetNetwork() {
        // Reset all neurons
        const neurons = this.svg.querySelectorAll('.neuron');
        neurons.forEach(n => {
            n.setAttribute('fill', this.config.colors.neuronInactive);
            n.setAttribute('stroke', '#adb5bd');
            n.setAttribute('stroke-width', '2');
        });

        // Reset all connections
        const connections = this.svg.querySelectorAll('.connection');
        connections.forEach(c => {
            c.setAttribute('stroke', this.config.colors.connectionInactive);
            c.setAttribute('stroke-width', '1.5');
            c.setAttribute('stroke-opacity', '0.3');
            c.style.animation = '';
            c.style.strokeDasharray = '';
            c.style.strokeDashoffset = '';
        });

        // Hide all values
        const values = this.svg.querySelectorAll('.neuron-value');
        values.forEach(v => {
            v.style.display = 'none';
        });
    }

    getNodePosition(layer, node) {
        return this.nodePositions[layer][node];
    }
}
