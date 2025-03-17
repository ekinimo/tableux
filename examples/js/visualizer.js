/**
 * Visualizer module - Handles integration with WASM Tableaux module
 */

// State object to track visualization state
let visualizerState = {
    isFormulaParsed: false,
    historyIndex: -1,
    history: [],
    autoStepInterval: null
};

/**
 * Initialize the visualizer module
 * @returns {Promise<Object|null>} The initialized visualizer or null if failed
 */
export async function initVisualizer() {
    try {
        // Import WASM module
        const module = await import('../pkg/tableux.js');
        await module.default();
        const { TableuxVisualizer } = module;
        
        // Create new visualizer instance
        const visualizer = new TableuxVisualizer();
        
        console.log("Tableux visualizer initialized successfully");
        return {
            instance: visualizer,
            state: visualizerState,
            parseFormula,
            step,
            stepBack,
            autoStep,
            parseAndProve,
            reset,
            generateSvg,
            isComplete: () => visualizer.is_complete(),
            isTautology: () => visualizer.is_tautology()
        };
    } catch (e) {
        console.error("Failed to load WASM module:", e);
        return null;
    }
}

/**
 * Parse a formula and set up the visualization
 * @param {Object} visualizer - The visualizer instance
 * @param {string} formula - The formula to parse
 * @returns {Promise<boolean>} True if parsing was successful
 */
async function parseFormula(visualizer, formula) {
    try {
        if (!formula.trim()) {
            throw new Error("Please enter a formula");
        }
        
        // Parse the formula
        const result = await visualizer.instance.parse_formula(formula.trim());
        
        // Reset state
        visualizerState.isFormulaParsed = true;
        visualizerState.history = [visualizer.instance.generate_svg()];
        visualizerState.historyIndex = 0;
        
        return true;
    } catch (e) {
        // If error is a string, it's from Rust
        if (typeof e === 'string') {
            throw new Error(e);
        }
        // Otherwise, it's a JS error
        throw e;
    }
}

/**
 * Step forward in the visualization
 * @param {Object} visualizer - The visualizer instance
 * @param {number} steps - Number of steps to take
 * @returns {boolean} True if progress was made
 */
function step(visualizer, steps = 1) {
    if (!visualizerState.isFormulaParsed) {
        throw new Error("Please parse a formula first");
    }
    
    if (isNaN(steps) || steps < 1) {
        throw new Error("Please enter a valid step count");
    }
    
    const madeProgress = visualizer.instance.step(steps);
    
    // Add to history if we've moved beyond the current point
    if (visualizerState.historyIndex < visualizerState.history.length - 1) {
        visualizerState.history = visualizerState.history.slice(0, visualizerState.historyIndex + 1);
    }
    
    // Add new state to history
    visualizerState.history.push(visualizer.instance.generate_svg());
    visualizerState.historyIndex = visualizerState.history.length - 1;
    
    return madeProgress;
}

/**
 * Step backward in the visualization history
 * @param {Object} visualizer - The visualizer instance
 * @returns {boolean} True if step back was successful
 */
function stepBack(visualizer) {
    if (visualizerState.historyIndex > 0) {
        visualizerState.historyIndex--;
        return true;
    }
    return false;
}

/**
 * Toggle auto-stepping
 * @param {Object} visualizer - The visualizer instance
 * @param {number} steps - Number of steps to take each interval
 * @param {function} onStep - Callback when a step is taken
 * @param {function} onStop - Callback when auto-stepping stops
 * @returns {boolean} True if auto-stepping was started, false if stopped
 */
function autoStep(visualizer, steps, onStep, onStop) {
    if (visualizerState.autoStepInterval) {
        // Stop auto-stepping
        clearInterval(visualizerState.autoStepInterval);
        visualizerState.autoStepInterval = null;
        return false;
    }
    
    if (!visualizerState.isFormulaParsed) {
        throw new Error("Please parse a formula first");
    }
    
    // Start auto-stepping
    visualizerState.autoStepInterval = setInterval(() => {
        try {
            if (visualizer.instance.is_complete()) {
                clearInterval(visualizerState.autoStepInterval);
                visualizerState.autoStepInterval = null;
                if (onStop) onStop();
                return;
            }
            
            const madeProgress = step(visualizer, steps);
            
            if (onStep) onStep(madeProgress);
            
            if (!madeProgress && !visualizer.instance.is_complete()) {
                clearInterval(visualizerState.autoStepInterval);
                visualizerState.autoStepInterval = null;
                throw new Error("Auto-step stopped: No progress made");
            }
        } catch (e) {
            clearInterval(visualizerState.autoStepInterval);
            visualizerState.autoStepInterval = null;
            if (onStop) onStop(e);
        }
    }, 300); // Step every 300ms
    
    return true;
}

/**
 * Parse and prove in one step
 * @param {Object} visualizer - The visualizer instance
 * @param {string} formula - The formula to parse
 * @param {number} steps - Maximum number of steps
 * @returns {string} The SVG output
 */
function parseAndProve(visualizer, formula, steps) {
    if (!formula.trim()) {
        throw new Error("Please enter a formula");
    }
    
    if (isNaN(steps) || steps < 1) {
        throw new Error("Please enter a valid step count");
    }
    
    const svgOutput = visualizer.instance.parse_and_prove(formula.trim(), steps);
    
    // If the output starts with "Parse error:", it's an error message
    if (typeof svgOutput === 'string' && svgOutput.startsWith('Parse error:')) {
        throw new Error(svgOutput);
    }
    
    // Update state
    visualizerState.isFormulaParsed = true;
    visualizerState.history = [svgOutput];
    visualizerState.historyIndex = 0;
    
    return svgOutput;
}

/**
 * Reset the visualizer
 * @param {Object} visualizer - The visualizer instance
 */
function reset(visualizer) {
    // Stop any auto-stepping
    if (visualizerState.autoStepInterval) {
        clearInterval(visualizerState.autoStepInterval);
        visualizerState.autoStepInterval = null;
    }
    
    // Reset state
    visualizerState.isFormulaParsed = false;
    visualizerState.historyIndex = -1;
    visualizerState.history = [];
    
    // Create a new visualizer instance
    visualizer.instance = new visualizer.constructor();
}

/**
 * Generate SVG from current state or history
 * @param {Object} visualizer - The visualizer instance
 * @returns {string} The SVG output
 */
function generateSvg(visualizer) {
    if (visualizerState.historyIndex >= 0 && visualizerState.historyIndex < visualizerState.history.length) {
        // Use stored history
        return visualizerState.history[visualizerState.historyIndex];
    } else {
        // Generate fresh
        return visualizer.instance.generate_svg();
    }
}
