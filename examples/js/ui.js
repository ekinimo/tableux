/**
 * UI module - Handles user interface interactions
 */

// UI elements
let elements = {};
let visualizer;

/**
 * Initialize UI components and attach event listeners
 * @param {Object} visualizerInstance - The visualizer instance
 */
export function initUI(visualizerInstance) {
    visualizer = visualizerInstance;
    
    // Store references to DOM elements
    cacheElements();
    
    // Make sure loading indicator is hidden at startup
    hideLoading();
    
    // Attach event listeners
    attachEventListeners();
    
    // Initialize help section toggle
    initHelpToggle();
    
    console.log("UI initialized successfully");
}

/**
 * Cache references to DOM elements
 */
function cacheElements() {
    elements = {
        formula: document.getElementById('formula'),
        output: document.getElementById('output'),
        error: document.getElementById('error'),
        loading: document.getElementById('loading'),
        status: document.getElementById('status'),
        
        // Buttons
        parseBtn: document.getElementById('parse-btn'),
        stepBtn: document.getElementById('step-btn'),
        stepBackBtn: document.getElementById('step-back-btn'),
        autoBtn: document.getElementById('auto-btn'),
        visualizeBtn: document.getElementById('visualize'),
        clearBtn: document.getElementById('clear'),
        exampleBtns: document.querySelectorAll('.example-btn'),
        
        // Inputs
        stepCount: document.getElementById('step-count'),
        steps: document.getElementById('steps'),
        
        // Help
        toggleHelpBtn: document.getElementById('toggle-help-btn'),
        helpContent: document.getElementById('help-content')
    };
}

/**
 * Attach event listeners to UI elements
 */
function attachEventListeners() {
    // Parse button
    elements.parseBtn.addEventListener('click', handleParse);
    
    // Step buttons
    elements.stepBtn.addEventListener('click', handleStep);
    elements.stepBackBtn.addEventListener('click', handleStepBack);
    elements.autoBtn.addEventListener('click', handleAutoStep);
    
    // Visualize and clear buttons
    elements.visualizeBtn.addEventListener('click', handleVisualize);
    elements.clearBtn.addEventListener('click', handleClear);
    
    // Example buttons
    elements.exampleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            elements.formula.value = btn.dataset.formula;
        });
    });
}

/**
 * Initialize the help section toggle
 */
function initHelpToggle() {
    elements.toggleHelpBtn.addEventListener('click', () => {
        elements.helpContent.classList.toggle('hidden');
        elements.toggleHelpBtn.classList.toggle('active');
    });
}

/**
 * Handle Parse button click
 */
async function handleParse() {
    try {
        hideError();
        showLoading();
        disableButton(elements.parseBtn);
        
        const formula = elements.formula.value;
        
        // Use the correct function name with snake_case
        const result = await visualizer.instance.parse_formula(formula.trim());
        
        // Check if parsing was successful
        if (result && typeof result === 'string' && result.startsWith('Parse error:')) {
            throw new Error(result);
        }
        
        // Update the state to reflect successful parsing
        visualizer.state.isFormulaParsed = true;
        visualizer.state.history = [visualizer.instance.generate_svg()];
        visualizer.state.historyIndex = 0;
        
        renderVisualization();
        updateButtonStates();
        
    } catch (e) {
        showError(e.message);
    } finally {
        // Always hide the loading indicator and enable the button
        hideLoading();
        enableButton(elements.parseBtn);
    }
}

/**
 * Handle Step forward button click
 */
function handleStep() {
    try {
        hideError();
        showLoading(); // Show loading indicator for step operation too
        
        const steps = parseInt(elements.stepCount.value, 10);
        
        if (isNaN(steps) || steps < 1) {
            throw new Error("Please enter a valid step count");
        }
        
        // Use the correct function name with snake_case
        const madeProgress = visualizer.instance.step(steps);
        
        // Update history if we've moved beyond the current point
        if (visualizer.state.historyIndex < visualizer.state.history.length - 1) {
            visualizer.state.history = visualizer.state.history.slice(0, visualizer.state.historyIndex + 1);
        }
        
        // Add new state to history
        visualizer.state.history.push(visualizer.instance.generate_svg());
        visualizer.state.historyIndex = visualizer.state.history.length - 1;
        
        renderVisualization();
        updateButtonStates();
        
        if (!madeProgress && !visualizer.instance.is_complete()) {
            showError('No progress made in this step');
        }
    } catch (e) {
        showError(e.message);
    } finally {
        // Always hide loading when done
        hideLoading();
    }
}

/**
 * Handle Step back button click
 */
function handleStepBack() {
    try {
        hideError();
        showLoading(); // Show loading for consistency
        
        // No need to call WASM for this operation, just navigate the history
        if (visualizer.state.historyIndex > 0) {
            visualizer.state.historyIndex--;
            renderVisualization();
            updateButtonStates();
        }
    } catch (e) {
        showError(e.message);
    } finally {
        hideLoading();
    }
}

/**
 * Handle Auto-step button click
 */
function handleAutoStep() {
    try {
        hideError();
        showLoading();
        
        const steps = parseInt(elements.stepCount.value, 10);
        
        if (isNaN(steps) || steps < 1) {
            throw new Error("Please enter a valid step count");
        }
        
        // Check if already auto-stepping
        if (visualizer.state.autoStepInterval) {
            // Stop auto-stepping
            clearInterval(visualizer.state.autoStepInterval);
            visualizer.state.autoStepInterval = null;
            elements.autoBtn.textContent = "Auto-step";
            hideLoading();
            return;
        }
        
        // Check if we can auto-step
        if (!visualizer.state.isFormulaParsed) {
            throw new Error("Please parse a formula first");
        }
        
        // Start auto-stepping
        elements.autoBtn.textContent = "Stop";
        
        visualizer.state.autoStepInterval = setInterval(() => {
            try {
                if (visualizer.instance.is_complete()) {
                    clearInterval(visualizer.state.autoStepInterval);
                    visualizer.state.autoStepInterval = null;
                    elements.autoBtn.textContent = "Auto-step";
                    hideLoading();
                    updateButtonStates();
                    return;
                }
                
                const madeProgress = handleStepForAutoStep(steps);
                
                if (!madeProgress && !visualizer.instance.is_complete()) {
                    clearInterval(visualizer.state.autoStepInterval);
                    visualizer.state.autoStepInterval = null;
                    elements.autoBtn.textContent = "Auto-step";
                    showError('Auto-step stopped: No progress made');
                    hideLoading();
                    updateButtonStates();
                }
            } catch (e) {
                clearInterval(visualizer.state.autoStepInterval);
                visualizer.state.autoStepInterval = null;
                elements.autoBtn.textContent = "Auto-step";
                showError(`Auto-step error: ${e.message}`);
                hideLoading();
                updateButtonStates();
            }
        }, 300); // Step every 300ms
        
    } catch (e) {
        showError(e.message);
        hideLoading();
    }
}

/**
 * Helper function for auto-stepping
 */
function handleStepForAutoStep(steps) {
    try {
        // Use the correct function name with snake_case
        const madeProgress = visualizer.instance.step(steps);
        
        // Update history if we've moved beyond the current point
        if (visualizer.state.historyIndex < visualizer.state.history.length - 1) {
            visualizer.state.history = visualizer.state.history.slice(0, visualizer.state.historyIndex + 1);
        }
        
        // Add new state to history
        visualizer.state.history.push(visualizer.instance.generate_svg());
        visualizer.state.historyIndex = visualizer.state.history.length - 1;
        
        renderVisualization();
        updateButtonStates();
        
        return madeProgress;
    } catch (e) {
        throw e;
    }
}

/**
 * Handle Visualize button click (run full proof)
 */
function handleVisualize() {
    try {
        hideError();
        showLoading();
        disableButton(elements.visualizeBtn);
        
        const formula = elements.formula.value;
        const steps = parseInt(elements.steps.value, 10);
        
        // Use the correct method name: instance.parse_and_prove
        const svgOutput = visualizer.instance.parse_and_prove(formula, steps);
        
        // Check if the output starts with "Parse error:"
        if (typeof svgOutput === 'string' && svgOutput.startsWith('Parse error:')) {
            throw new Error(svgOutput);
        }
        
        // Update visualization
        elements.output.innerHTML = svgOutput;
        
        // Update visualizer state to reflect successful parsing
        visualizer.state.isFormulaParsed = true;
        visualizer.state.history = [svgOutput];
        visualizer.state.historyIndex = 0;
        
        updateStatus();
        updateButtonStates();
        
    } catch (e) {
        showError(e.message);
    } finally {
        // Always hide loading and enable button, even on error
        hideLoading();
        enableButton(elements.visualizeBtn);
    }
}

/**
 * Handle Clear button click
 */
function handleClear() {
    try {
        // Stop any auto-stepping
        if (visualizer.state.autoStepInterval) {
            clearInterval(visualizer.state.autoStepInterval);
            visualizer.state.autoStepInterval = null;
            elements.autoBtn.textContent = "Auto-step";
        }
        
        // Reset state
        visualizer.state.isFormulaParsed = false;
        visualizer.state.historyIndex = -1;
        visualizer.state.history = [];
        
        // Create a new visualizer instance
        visualizer.instance = new visualizer.instance.constructor();
        
        // Clear output
        elements.output.innerHTML = '';
        
        // Hide status and error
        elements.status.classList.add('hidden');
        hideError();
        hideLoading();
        
        // Update button states
        updateButtonStates();
    } catch (e) {
        showError(`Clear error: ${e.message}`);
    }
}

/**
 * Render the current visualization state
 */
function renderVisualization() {
    try {
        let svgString;
        
        if (visualizer.state.historyIndex >= 0 && 
            visualizer.state.historyIndex < visualizer.state.history.length) {
            // Use stored history
            svgString = visualizer.state.history[visualizer.state.historyIndex];
        } else {
            // Generate fresh
            svgString = visualizer.instance.generate_svg();
        }
        
        elements.output.innerHTML = svgString;
        updateStatus();
    } catch (e) {
        showError(`Rendering error: ${e.message}`);
    } finally {
        // Always ensure loading is hidden after rendering
        hideLoading();
    }
}

/**
 * Update the status indicator based on current state
 */
function updateStatus() {
    const status = elements.status;
    status.classList.remove('hidden', 'success', 'warning');
    
    if (visualizer.instance.is_tautology()) {
        status.textContent = "TAUTOLOGY";
        status.classList.add('success');
    } else if (visualizer.instance.is_complete()) {
        status.textContent = "NOT COMPLETE";
        status.classList.add('warning');
    } else {
        status.textContent = "IN PROGRESS";
        status.classList.add('warning');
    }
    
    status.classList.remove('hidden');
}

/**
 * Update button enabled/disabled states based on current state
 */
function updateButtonStates() {
    const state = visualizer.state;
    
    // Step buttons
    elements.stepBackBtn.disabled = state.historyIndex <= 0;
    elements.stepBtn.disabled = !state.isFormulaParsed || visualizer.instance.is_complete();
    elements.autoBtn.disabled = !state.isFormulaParsed || visualizer.instance.is_complete();
    
    // If auto-stepping is active, disable step buttons
    if (state.autoStepInterval) {
        elements.stepBtn.disabled = true;
        elements.stepBackBtn.disabled = true;
    }
}

/**
 * Show error message
 * @param {string} message - The error message to display
 */
function showError(message) {
    hideLoading(); // Always hide loading when showing an error
    elements.error.textContent = message;
    elements.error.classList.remove('hidden');
}

/**
 * Hide error message
 */
function hideError() {
    elements.error.textContent = '';
    elements.error.classList.add('hidden');
}

/**
 * Show loading indicator
 */
function showLoading() {
    elements.loading.classList.remove('hidden');
    elements.loading.classList.add('flex'); // Using flex display for alignment
}

/**
 * Hide loading indicator
 */
function hideLoading() {
    elements.loading.classList.add('hidden');
    elements.loading.classList.remove('flex');
}

/**
 * Disable a button
 * @param {HTMLElement} button - The button to disable
 */
function disableButton(button) {
    button.disabled = true;
}

/**
 * Enable a button
 * @param {HTMLElement} button - The button to enable
 */
function enableButton(button) {
    button.disabled = false;
}
