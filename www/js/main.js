import { initUI } from './ui.js';
import { initVisualizer } from './visualizer.js';
import { setupDownloader } from './downloader.js';

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', async () => {
    try {
        // Show loading indicator
        const loadingElement = document.getElementById('loading');
        if (loadingElement) {
            loadingElement.classList.remove('hidden');
        }
        
        // Initialize main components
        const visualizer = await initVisualizer();
        
        // Check if visualizer was loaded successfully
        if (!visualizer) {
            showError("Failed to initialize visualizer. Please ensure WebAssembly is supported by your browser.");
            hideLoading();
            return;
        }
        
        // Initialize UI with visualizer instance
        initUI(visualizer);
        
        // Setup SVG download functionality
        setupDownloader(visualizer);
        
        // Setup GitHub link
        setupGitHubLink();
        
        // Setup fullscreen functionality
        setupFullscreen();
        
        // Hide loading indicator
        hideLoading();
        
        console.log("Application initialized successfully");
    } catch (error) {
        console.error("Initialization error:", error);
        showError(`Failed to initialize: ${error.message || "Unknown error loading WASM module"}`);
        hideLoading();
    }
});

// Show error message
function showError(message) {
    const errorElement = document.getElementById('error');
    if (errorElement) {
        errorElement.textContent = message;
        errorElement.classList.remove('hidden');
    }
}

// Hide loading indicator
function hideLoading() {
    const loadingElement = document.getElementById('loading');
    if (loadingElement) {
        loadingElement.classList.add('hidden');
    }
}

// Initialize theme based on user preference
initTheme();

function initTheme() {
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    const isDarkMode = localStorage.getItem('darkMode') === 'true';
    
    // Apply saved theme preference
    if (isDarkMode) {
        document.body.classList.add('dark-theme');
        updateThemeIcon(true);
    } else {
        document.body.classList.remove('dark-theme');
        updateThemeIcon(false);
    }
    
    // Add toggle event listener
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const isCurrentlyDark = document.body.classList.contains('dark-theme');
            
            if (isCurrentlyDark) {
                document.body.classList.remove('dark-theme');
                localStorage.setItem('darkMode', 'false');
            } else {
                document.body.classList.add('dark-theme');
                localStorage.setItem('darkMode', 'true');
            }
            
            updateThemeIcon(!isCurrentlyDark);
        });
    }
}

function updateThemeIcon(isDark) {
    const themeIcon = document.getElementById('theme-icon');
    if (!themeIcon) return;
    
    if (isDark) {
        // Moon icon for dark mode
        themeIcon.innerHTML = `
            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
        `;
    } else {
        // Sun icon for light mode
        themeIcon.innerHTML = `
            <circle cx="12" cy="12" r="5"></circle>
            <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"></path>
        `;
    }
}

// Setup GitHub link functionality
function setupGitHubLink() {
    const githubLink = document.querySelector('.github-link');
    if (githubLink) {
        githubLink.addEventListener('click', (e) => {
            e.preventDefault();
            window.open('https://github.com/ekinimo/tableux', '_blank');
        });
    }
}

// Setup fullscreen functionality
function setupFullscreen() {
    const fullscreenBtn = document.getElementById('fullscreen-btn');
    const visualizationContainer = document.querySelector('.visualization-container');
    
    if (!fullscreenBtn || !visualizationContainer) return;
    
    let isFullscreen = false;
    
    fullscreenBtn.addEventListener('click', () => {
        isFullscreen = !isFullscreen;
        
        if (isFullscreen) {
            // Enter fullscreen
            visualizationContainer.classList.add('fullscreen-mode');
            fullscreenBtn.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3"></path>
                </svg>
                Exit Fullscreen
            `;
            
            // Fix SVG in fullscreen mode
            const svg = document.querySelector('#output svg');
            if (svg) {
                enhanceSvg(svg, true);
            }
        } else {
            // Exit fullscreen
            visualizationContainer.classList.remove('fullscreen-mode');
            fullscreenBtn.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path>
                </svg>
                Fullscreen
            `;
            
            // Reset SVG in normal mode
            const svg = document.querySelector('#output svg');
            if (svg) {
                enhanceSvg(svg, false);
            }
        }
    });
    
    // Handle ESC key to exit fullscreen
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && isFullscreen) {
            fullscreenBtn.click();
        }
    });
    
    // Observe changes to the output element to enhance SVG when it's added
    const outputElement = document.getElementById('output');
    if (outputElement) {
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    // Find SVG elements that were added
                    mutation.addedNodes.forEach((node) => {
                        if (node.tagName && node.tagName.toLowerCase() === 'svg') {
                            enhanceSvg(node, isFullscreen);
                        }
                    });
                }
            });
        });
        
        observer.observe(outputElement, { childList: true });
    }
}

// Enhance SVG for better display
function enhanceSvg(svg, isFullscreen) {
    if (!svg) return;
    
    // Store original dimensions if not already stored
    if (!svg.dataset.originalWidth && svg.hasAttribute('width')) {
        svg.dataset.originalWidth = svg.getAttribute('width');
    }
    if (!svg.dataset.originalHeight && svg.hasAttribute('height')) {
        svg.dataset.originalHeight = svg.getAttribute('height');
    }
    
    // Ensure SVG has a viewBox for proper scaling
    if (!svg.hasAttribute('viewBox') && 
        svg.dataset.originalWidth && 
        svg.dataset.originalHeight) {
        svg.setAttribute('viewBox', `0 0 ${svg.dataset.originalWidth} ${svg.dataset.originalHeight}`);
    }
    
    // Add CSS class for styling
    svg.classList.add('responsive-svg');
    
    if (isFullscreen) {
        // Optimize for fullscreen
        svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
        svg.removeAttribute('width');
        svg.removeAttribute('height');
    } else {
        // Standard view - let CSS handle responsive sizing
        svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
        svg.setAttribute('width', '100%');
        svg.removeAttribute('height');
    }
    
    // Fix for Firefox text alignment
    if (navigator.userAgent.toLowerCase().indexOf('firefox') > -1) {
        const textElements = svg.querySelectorAll('text');
        textElements.forEach(text => {
            if (text.hasAttribute('dominant-baseline') && 
                text.getAttribute('dominant-baseline') === 'middle') {
                text.setAttribute('dy', '0.3em');
            }
        });
    }
}
