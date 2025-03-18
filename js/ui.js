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
    if (elements.parseBtn) {
        elements.parseBtn.addEventListener('click', handleParse);
    }
    
    // Step buttons
    if (elements.stepBtn) {
        elements.stepBtn.addEventListener('click', handleStep);
    }
    
    if (elements.stepBackBtn) {
        elements.stepBackBtn.addEventListener('click', handleStepBack);
    }
    
    if (elements.autoBtn) {
        elements.autoBtn.addEventListener('click', handleAutoStep);
    }
    
    // Visualize and clear buttons
    if (elements.visualizeBtn) {
        elements.visualizeBtn.addEventListener('click', handleVisualize);
    }
    
    if (elements.clearBtn) {
        elements.clearBtn.addEventListener('click', handleClear);
    }
    
    // Example buttons
    if (elements.exampleBtns) {
        elements.exampleBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                if (elements.formula) {
                    elements.formula.value = btn.dataset.formula;
                }
            });
        });
    }
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', handleKeyboard);
}

/**
 * Handle keyboard shortcuts
 */
function handleKeyboard(e) {
    // Don't trigger shortcuts when typing in an input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        return;
    }
    
    switch (e.key) {
        case 'p':
            if (elements.parseBtn && !elements.parseBtn.disabled) {
                elements.parseBtn.click();
            }
            break;
        case 'n':
            if (elements.stepBtn && !elements.stepBtn.disabled) {
                elements.stepBtn.click();
            }
            break;
        case 'b':
            if (elements.stepBackBtn && !elements.stepBackBtn.disabled) {
                elements.stepBackBtn.click();
            }
            break;
        case 'r':
            if (elements.visualizeBtn && !elements.visualizeBtn.disabled) {
                elements.visualizeBtn.click();
            }
            break;
        case 'c':
            if (elements.clearBtn && !elements.clearBtn.disabled) {
                elements.clearBtn.click();
            }
            break;
    }
}

/**
 * Initialize the help section toggle
 */
function initHelpToggle() {
    if (elements.toggleHelpBtn && elements.helpContent) {
        elements.toggleHelpBtn.addEventListener('click', () => {
            elements.helpContent.classList.toggle('hidden');
            elements.toggleHelpBtn.classList.toggle('active');
        });
    }
}

/**
 * Handle Parse button click
 */
async function handleParse() {
    try {
        hideError();
        showLoading();
        
        if (elements.parseBtn) {
            disableButton(elements.parseBtn);
        }
        
        const formula = elements.formula ? elements.formula.value.trim() : '';
        
        if (!formula) {
            throw new Error("Please enter a formula");
        }
        
        // Parse the formula
        const result = await visualizer.instance.parse_formula(formula);
        
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
        if (elements.parseBtn) {
            enableButton(elements.parseBtn);
        }
    }
}

/**
 * Handle Step forward button click
 */
function handleStep() {
    try {
        hideError();
        showLoading();
        
        const steps = parseInt(elements.stepCount ? elements.stepCount.value : 1, 10);
        
        if (isNaN(steps) || steps < 1) {
            throw new Error("Please enter a valid step count");
        }
        
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
        hideLoading();
    }
}

/**
 * Handle Step back button click
 */
function handleStepBack() {
    try {
        hideError();
        showLoading();
        
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
        
        const steps = parseInt(elements.stepCount ? elements.stepCount.value : 1, 10);
        
        if (isNaN(steps) || steps < 1) {
            throw new Error("Please enter a valid step count");
        }
        
        // Check if already auto-stepping
        if (visualizer.state.autoStepInterval) {
            // Stop auto-stepping
            clearInterval(visualizer.state.autoStepInterval);
            visualizer.state.autoStepInterval = null;
            
            if (elements.autoBtn) {
                elements.autoBtn.textContent = "Auto-step";
            }
            
            hideLoading();
            return;
        }
        
        // Check if we can auto-step
        if (!visualizer.state.isFormulaParsed) {
            throw new Error("Please parse a formula first");
        }
        
        // Start auto-stepping
        if (elements.autoBtn) {
            elements.autoBtn.textContent = "Stop";
        }
        
        visualizer.state.autoStepInterval = setInterval(() => {
            try {
                if (visualizer.instance.is_complete()) {
                    clearInterval(visualizer.state.autoStepInterval);
                    visualizer.state.autoStepInterval = null;
                    
                    if (elements.autoBtn) {
                        elements.autoBtn.textContent = "Auto-step";
                    }
                    
                    hideLoading();
                    updateButtonStates();
                    return;
                }
                
                const madeProgress = handleStepForAutoStep(steps);
                
                if (!madeProgress && !visualizer.instance.is_complete()) {
                    clearInterval(visualizer.state.autoStepInterval);
                    visualizer.state.autoStepInterval = null;
                    
                    if (elements.autoBtn) {
                        elements.autoBtn.textContent = "Auto-step";
                    }
                    
                    showError('Auto-step stopped: No progress made');
                    hideLoading();
                    updateButtonStates();
                }
            } catch (e) {
                clearInterval(visualizer.state.autoStepInterval);
                visualizer.state.autoStepInterval = null;
                
                if (elements.autoBtn) {
                    elements.autoBtn.textContent = "Auto-step";
                }
                
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
        
        if (elements.visualizeBtn) {
            disableButton(elements.visualizeBtn);
        }
        
        const formula = elements.formula ? elements.formula.value.trim() : '';
        
        if (!formula) {
            throw new Error("Please enter a formula");
        }
        
        const steps = parseInt(elements.steps ? elements.steps.value : 100, 10);
        
        if (isNaN(steps) || steps < 1) {
            throw new Error("Please enter a valid step count");
        }
        
        // Use the correct method name: instance.parse_and_prove
        const svgOutput = visualizer.instance.parse_and_prove(formula, steps);
        
        // Check if the output starts with "Parse error:"
        if (typeof svgOutput === 'string' && svgOutput.startsWith('Parse error:')) {
            throw new Error(svgOutput);
        }
        
        // Update visualization
        if (elements.output) {
            elements.output.innerHTML = svgOutput;
            
            // Enhance SVG for better display
            const svg = elements.output.querySelector('svg');
            if (svg) {
                enhanceSvg(svg);
            }
        }
        
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
        if (elements.visualizeBtn) {
            enableButton(elements.visualizeBtn);
        }
    }
}

/**
 * Enhance SVG for better display
 * @param {SVGElement} svg - SVG element to enhance
 */
function enhanceSvg(svg) {
    if (!svg) return;
    
    // Add CSS class for responsive sizing
    svg.classList.add('responsive-svg');
    
    // Set attributes for better display
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', 'auto');
    
    // Add viewBox if missing
    if (!svg.hasAttribute('viewBox') && 
        svg.hasAttribute('width') && 
        svg.hasAttribute('height')) {
        
        const width = svg.getAttribute('width');
        const height = svg.getAttribute('height');
        
        svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    }
    
    // Set preserveAspectRatio for better scaling
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
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
            
            if (elements.autoBtn) {
                elements.autoBtn.textContent = "Auto-step";
            }
        }
        
        // Reset state
        visualizer.state.isFormulaParsed = false;
        visualizer.state.historyIndex = -1;
        visualizer.state.history = [];
        
        // Create a new visualizer instance
        visualizer.instance = new visualizer.instance.constructor();
        
        // Clear output
        if (elements.output) {
            elements.output.innerHTML = '';
        }
        
        // Hide status and error
        if (elements.status) {
            elements.status.classList.add('hidden');
        }
        
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
        if (!elements.output) return;
        
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
        
        // Enhance SVG for better display
        const svg = elements.output.querySelector('svg');
        if (svg) {
            enhanceSvg(svg);
        }
        
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
    if (!elements.status) return;
    
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
    if (elements.stepBackBtn) {
        elements.stepBackBtn.disabled = state.historyIndex <= 0;
    }
    
    if (elements.stepBtn) {
        elements.stepBtn.disabled = !state.isFormulaParsed || visualizer.instance.is_complete();
    }
    
    if (elements.autoBtn) {
        elements.autoBtn.disabled = !state.isFormulaParsed || visualizer.instance.is_complete();
    }
    
    // If auto-stepping is active, disable step buttons
    if (state.autoStepInterval) {
        if (elements.stepBtn) {
            elements.stepBtn.disabled = true;
        }
        
        if (elements.stepBackBtn) {
            elements.stepBackBtn.disabled = true;
        }
    }
}

/**
 * Show error message
 * @param {string} message - The error message to display
 */
function showError(message) {
    hideLoading(); // Always hide loading when showing an error
    
    if (!elements.error) return;
    
    elements.error.textContent = message;
    elements.error.classList.remove('hidden');
}

/**
 * Hide error message
 */
function hideError() {
    if (!elements.error) return;
    
    elements.error.textContent = '';
    elements.error.classList.add('hidden');
}

/**
 * Show loading indicator
 */
function showLoading() {
    if (!elements.loading) return;
    
    elements.loading.classList.remove('hidden');
}

/**
 * Hide loading indicator
 */
function hideLoading() {
    if (!elements.loading) return;
    
    elements.loading.classList.add('hidden');
}

/**
 * Disable a button
 * @param {HTMLElement} button - The button to disable
 */
function disableButton(button) {
    if (button) {
        button.disabled = true;
    }
}

/**
 * Enable a button
 * @param {HTMLElement} button - The button to enable
 */
function enableButton(button) {
    if (button) {
        button.disabled = false;
    }
}
