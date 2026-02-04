/**
 * activationPlot.js
 * Draws activation function plots on canvas
 */

class ActivationPlotDrawer {
    constructor(canvasId, mlpData) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.data = mlpData;

        this.config = {
            padding: 60,
            gridColor: '#e9ecef',
            axisColor: '#495057',
            curveColor: '#0d6efd',
            pointColor: '#ffc107',
            textColor: '#212529'
        };
    }

    drawReLU(highlightZ = null, highlightA = null) {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        const padding = this.config.padding;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Set up coordinate system
        const plotWidth = width - 2 * padding;
        const plotHeight = height - 2 * padding;
        const centerX = padding + plotWidth / 2;
        const centerY = padding + plotHeight / 2;

        // Define range
        const zMin = -3;
        const zMax = 3;
        const aMin = 0;
        const aMax = 3;

        const scaleX = plotWidth / (zMax - zMin);
        const scaleY = plotHeight / (aMax - aMin);

        // Helper function to convert data coordinates to canvas coordinates
        const toCanvasX = (z) => centerX + (z * scaleX);
        const toCanvasY = (a) => padding + plotHeight - (a * scaleY);

        // Draw grid
        this.drawGrid(ctx, padding, plotWidth, plotHeight, toCanvasX, toCanvasY);

        // Draw axes
        this.drawAxes(ctx, centerX, centerY, padding, plotWidth, plotHeight);

        // Draw ReLU curve
        ctx.beginPath();
        ctx.strokeStyle = this.config.curveColor;
        ctx.lineWidth = 3;

        // Left part (z < 0, output = 0)
        ctx.moveTo(toCanvasX(zMin), toCanvasY(0));
        ctx.lineTo(toCanvasX(0), toCanvasY(0));

        // Right part (z >= 0, output = z)
        ctx.lineTo(toCanvasX(zMax), toCanvasY(zMax));
        ctx.stroke();

        // Draw highlighted point if provided
        if (highlightZ !== null && highlightA !== null) {
            // Draw vertical line from z-axis to point
            ctx.beginPath();
            ctx.strokeStyle = this.config.pointColor;
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.moveTo(toCanvasX(highlightZ), toCanvasY(0));
            ctx.lineTo(toCanvasX(highlightZ), toCanvasY(highlightA));
            ctx.stroke();

            // Draw horizontal line from y-axis to point
            ctx.beginPath();
            ctx.moveTo(toCanvasX(zMin), toCanvasY(highlightA));
            ctx.lineTo(toCanvasX(highlightZ), toCanvasY(highlightA));
            ctx.stroke();
            ctx.setLineDash([]);

            // Draw the point
            ctx.beginPath();
            ctx.arc(toCanvasX(highlightZ), toCanvasY(highlightA), 8, 0, Math.PI * 2);
            ctx.fillStyle = this.config.pointColor;
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Draw point label
            ctx.fillStyle = this.config.textColor;
            ctx.font = 'bold 14px Arial';
            ctx.textAlign = 'left';
            ctx.fillText(
                `(${highlightZ.toFixed(2)}, ${highlightA.toFixed(2)})`,
                toCanvasX(highlightZ) + 15,
                toCanvasY(highlightA) - 10
            );
        }

        // Draw axis labels
        this.drawAxisLabels(ctx, centerX, centerY, padding, plotWidth, plotHeight);

        // Draw title
        ctx.fillStyle = this.config.textColor;
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('ReLU Activation Function', width / 2, 30);
    }

    drawGrid(ctx, padding, plotWidth, plotHeight, toCanvasX, toCanvasY) {
        ctx.strokeStyle = this.config.gridColor;
        ctx.lineWidth = 1;

        // Vertical grid lines
        for (let z = -3; z <= 3; z++) {
            ctx.beginPath();
            ctx.moveTo(toCanvasX(z), padding);
            ctx.lineTo(toCanvasX(z), padding + plotHeight);
            ctx.stroke();
        }

        // Horizontal grid lines
        for (let a = 0; a <= 3; a++) {
            ctx.beginPath();
            ctx.moveTo(padding, toCanvasY(a));
            ctx.lineTo(padding + plotWidth, toCanvasY(a));
            ctx.stroke();
        }
    }

    drawAxes(ctx, centerX, centerY, padding, plotWidth, plotHeight) {
        ctx.strokeStyle = this.config.axisColor;
        ctx.lineWidth = 2;

        // X-axis
        ctx.beginPath();
        ctx.moveTo(padding, centerY);
        ctx.lineTo(padding + plotWidth, centerY);
        ctx.stroke();

        // Y-axis
        ctx.beginPath();
        ctx.moveTo(centerX, padding);
        ctx.lineTo(centerX, padding + plotHeight);
        ctx.stroke();

        // Arrow heads
        this.drawArrow(ctx, padding + plotWidth, centerY, 'right');
        this.drawArrow(ctx, centerX, padding, 'up');
    }

    drawArrow(ctx, x, y, direction) {
        const size = 8;
        ctx.fillStyle = this.config.axisColor;
        ctx.beginPath();

        switch (direction) {
            case 'right':
                ctx.moveTo(x, y);
                ctx.lineTo(x - size, y - size / 2);
                ctx.lineTo(x - size, y + size / 2);
                break;
            case 'up':
                ctx.moveTo(x, y);
                ctx.lineTo(x - size / 2, y + size);
                ctx.lineTo(x + size / 2, y + size);
                break;
        }

        ctx.closePath();
        ctx.fill();
    }

    drawAxisLabels(ctx, centerX, centerY, padding, plotWidth, plotHeight) {
        ctx.fillStyle = this.config.textColor;
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';

        // X-axis labels
        for (let z = -3; z <= 3; z++) {
            const x = centerX + (z * (plotWidth / 6));
            ctx.fillText(z.toString(), x, centerY + 20);
        }

        // Y-axis labels
        ctx.textAlign = 'right';
        for (let a = 0; a <= 3; a++) {
            const y = padding + plotHeight - (a * (plotHeight / 3));
            ctx.fillText(a.toString(), centerX - 10, y + 5);
        }

        // Axis titles
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('z (pre-activation)', centerX, padding + plotHeight + 45);

        ctx.save();
        ctx.translate(padding - 45, centerY);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('a (activation)', 0, 0);
        ctx.restore();
    }

    drawSigmoid(highlightZ = null, highlightA = null) {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        const padding = this.config.padding;

        ctx.clearRect(0, 0, width, height);

        const plotWidth = width - 2 * padding;
        const plotHeight = height - 2 * padding;

        const zMin = -6;
        const zMax = 6;
        const scaleX = plotWidth / (zMax - zMin);
        const scaleY = plotHeight;

        const toCanvasX = (z) => padding + (z - zMin) * scaleX;
        const toCanvasY = (a) => padding + plotHeight - (a * plotHeight);

        const sigmoid = (z) => 1 / (1 + Math.exp(-z));

        // Draw curve
        ctx.beginPath();
        ctx.strokeStyle = this.config.curveColor;
        ctx.lineWidth = 3;

        for (let z = zMin; z <= zMax; z += 0.1) {
            const a = sigmoid(z);
            const x = toCanvasX(z);
            const y = toCanvasY(a);

            if (z === zMin) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();

        // Draw highlighted point if provided
        if (highlightZ !== null && highlightA !== null) {
            ctx.beginPath();
            ctx.arc(toCanvasX(highlightZ), toCanvasY(highlightA), 8, 0, Math.PI * 2);
            ctx.fillStyle = this.config.pointColor;
            ctx.fill();
        }

        // Draw title
        ctx.fillStyle = this.config.textColor;
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Sigmoid Activation Function', width / 2, 30);
    }
}
