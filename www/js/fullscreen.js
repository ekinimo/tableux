/**
 * Fullscreen handler for visualization
 */

let isFullscreen = false;

export function setupFullscreenButton() {
    const fullscreenBtn = document.getElementById('fullscreen-btn');
    const visualizationContainer = document.querySelector('.visualization-container');
    
    if (!fullscreenBtn || !visualizationContainer) return;
    
    fullscreenBtn.addEventListener('click', toggleFullscreen);
    
    // Add escape key handler for exiting fullscreen
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && isFullscreen) {
            exitFullscreen();
        }
    });
    
    // Create close button for fullscreen mode
    const closeBtn = document.createElement('button');
    closeBtn.className = 'fullscreen-close-btn';
    closeBtn.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
    `;
    closeBtn.style.display = 'none';
    closeBtn.addEventListener('click', exitFullscreen);
    closeBtn.id = 'fullscreen-close-btn';
    
    // Add close button to the container
    visualizationContainer.appendChild(closeBtn);
}

// Toggle fullscreen mode
function toggleFullscreen() {
    if (isFullscreen) {
        exitFullscreen();
    } else {
        enterFullscreen();
    }
}

// Enter fullscreen mode
function enterFullscreen() {
    const visualizationContainer = document.querySelector('.visualization-container');
    const fullscreenBtn = document.getElementById('fullscreen-btn');
    const closeBtn = document.getElementById('fullscreen-close-btn');
    
    if (!visualizationContainer || !fullscreenBtn) return;
    
    // Enter fullscreen mode
    visualizationContainer.classList.add('visualization-fullscreen');
    
    // Update button text
    fullscreenBtn.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3"></path>
        </svg>
        Exit Fullscreen
    `;
    
    // Show close button
    if (closeBtn) {
        closeBtn.style.display = 'block';
    }
    
    // Adapt SVG for fullscreen
    adaptSvgForFullscreen();
    
    // Update state
    isFullscreen = true;
    
    // Dispatch custom event for other modules
    document.dispatchEvent(new CustomEvent('enter-fullscreen'));
}

// Exit fullscreen mode
function exitFullscreen() {
    const visualizationContainer = document.querySelector('.visualization-container');
    const fullscreenBtn = document.getElementById('fullscreen-btn');
    const closeBtn = document.getElementById('fullscreen-close-btn');
    
    if (!visualizationContainer || !fullscreenBtn) return;
    
    // Exit fullscreen mode
    visualizationContainer.classList.remove('visualization-fullscreen');
    
    // Update button text
    fullscreenBtn.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path>
        </svg>
        Fullscreen
    `;
    
    // Hide close button
    if (closeBtn) {
        closeBtn.style.display = 'none';
    }
    
    // Reset SVG to normal view
    resetSvgFromFullscreen();
    
    // Update state
    isFullscreen = false;
    
    // Dispatch custom event for other modules
    document.dispatchEvent(new CustomEvent('exit-fullscreen'));
}

// Adapt SVG for fullscreen view
function adaptSvgForFullscreen() {
    const svg = document.querySelector('#output svg');
    if (!svg) return;
    
    // Store original attributes for later restoration
    svg.dataset.originalWidth = svg.getAttribute('width') || '';
    svg.dataset.originalHeight = svg.getAttribute('height') || '';
    svg.dataset.originalViewBox = svg.getAttribute('viewBox') || '';
    svg.dataset.originalTransform = svg.getAttribute('transform') || '';
    
    // Make SVG responsive to the fullscreen container
    svg.removeAttribute('width');
    svg.removeAttribute('height');
    
    // Ensure viewBox is set for proper scaling
    if (!svg.getAttribute('viewBox') && 
        svg.dataset.originalWidth && 
        svg.dataset.originalHeight) {
        svg.setAttribute('viewBox', `0 0 ${svg.dataset.originalWidth} ${svg.dataset.originalHeight}`);
    }
    
    // Set preserveAspectRatio for better display
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    
    // Reset any transforms
    svg.style.transform = 'scale(1)';
}

// Reset SVG from fullscreen to normal view
function resetSvgFromFullscreen() {
    const svg = document.querySelector('#output svg');
    if (!svg) return;
    
    // Restore original attributes
    if (svg.dataset.originalWidth) {
        svg.setAttribute('width', svg.dataset.originalWidth);
    }
    
    if (svg.dataset.originalHeight) {
        svg.setAttribute('height', svg.dataset.originalHeight);
    }
    
    if (svg.dataset.originalViewBox) {
        svg.setAttribute('viewBox', svg.dataset.originalViewBox);
    } else {
        // If no original viewBox, remove the one we added
        svg.removeAttribute('viewBox');
    }
    
    if (svg.dataset.originalTransform) {
        svg.setAttribute('transform', svg.dataset.originalTransform);
    } else {
        svg.removeAttribute('transform');
    }
    
    // Reset any inline styles
    svg.style.transform = '';
}
