/**
 * SVG Handler Module - Manages SVG sizing and rendering
 */

// Track current SVG element
let currentSvg = null;
let zoomLevel = 1;
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 2;
const ZOOM_STEP = 0.1;

// Initialize SVG handling
export function initSvgHandler() {
    // Setup zoom controls
    setupZoomControls();
    
    // Observe the output container for SVG changes
    observeSvgChanges();
    
    // Add resize listener for responsive adjustments
    window.addEventListener('resize', handleResize);
    
    console.log("SVG handler initialized");
}

// Setup zoom controls
function setupZoomControls() {
    // Create zoom controls container
    const zoomControls = document.createElement('div');
    zoomControls.className = 'zoom-controls';
    zoomControls.innerHTML = `
        <button id="zoom-in" class="zoom-btn" title="Zoom In">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                <line x1="11" y1="8" x2="11" y2="14"></line>
                <line x1="8" y1="11" x2="14" y2="11"></line>
            </svg>
        </button>
        <span id="zoom-level">100%</span>
        <button id="zoom-out" class="zoom-btn" title="Zoom Out">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                <line x1="8" y1="11" x2="14" y2="11"></line>
            </svg>
        </button>
        <button id="zoom-reset" class="zoom-btn" title="Reset Zoom">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M3 12a9 9 0 1 0 18 0 9 9 0 0 0-18 0z"></path>
                <path d="M14 8l-4 4-4-4"></path>
                <path d="M10 12V4"></path>
            </svg>
        </button>
    `;
    
    // Add controls to visualization container
    const container = document.querySelector('.visualization-container');
    if (container) {
        container.appendChild(zoomControls);
    }
    
    // Add event listeners
    document.getElementById('zoom-in')?.addEventListener('click', () => zoomSvg(ZOOM_STEP));
    document.getElementById('zoom-out')?.addEventListener('click', () => zoomSvg(-ZOOM_STEP));
    document.getElementById('zoom-reset')?.addEventListener('click', resetZoom);
    
    // Add wheel event listener to the visualization container for zooming
    container?.addEventListener('wheel', handleWheel, { passive: false });
}

// Handle wheel event for zooming
function handleWheel(e) {
    // Only zoom if Ctrl key is pressed (standard zoom gesture)
    if (e.ctrlKey) {
        e.preventDefault();
        const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP;
        zoomSvg(delta);
    }
}

// Apply zoom to SVG
function zoomSvg(delta) {
    if (!currentSvg) return;
    
    zoomLevel = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoomLevel + delta));
    
    // Apply zoom transformation
    currentSvg.style.transform = `scale(${zoomLevel})`;
    
    // Update zoom level display
    const zoomLevelDisplay = document.getElementById('zoom-level');
    if (zoomLevelDisplay) {
        zoomLevelDisplay.textContent = `${Math.round(zoomLevel * 100)}%`;
    }
}

// Reset zoom to 100%
function resetZoom() {
    if (!currentSvg) return;
    
    zoomLevel = 1;
    currentSvg.style.transform = 'scale(1)';
    
    // Update zoom level display
    const zoomLevelDisplay = document.getElementById('zoom-level');
    if (zoomLevelDisplay) {
        zoomLevelDisplay.textContent = '100%';
    }
}

// Observe SVG changes
function observeSvgChanges() {
    const outputContainer = document.getElementById('output');
    if (!outputContainer) return;
    
    // Create a MutationObserver to watch for SVG changes
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                // Check if an SVG was added
                mutation.addedNodes.forEach((node) => {
                    if (node.tagName && node.tagName.toLowerCase() === 'svg') {
                        // Process the new SVG
                        processSvg(node);
                    }
                });
            }
        });
    });
    
    // Start observing the output container
    observer.observe(outputContainer, { childList: true });
}

// Process a newly added SVG
function processSvg(svg) {
    // Store reference to current SVG
    currentSvg = svg;
    
    // Create a custom event to notify Firefox-specific fixes
    const svgLoadedEvent = new CustomEvent('svg-loaded', { 
        detail: { svg }
    });
    document.dispatchEvent(svgLoadedEvent);
    
    // Ensure SVG has proper viewBox
    ensureViewBox(svg);
    
    // Add basic pan & zoom if in fullscreen
    if (document.querySelector('.visualization-fullscreen')) {
        makeFullscreenSvgInteractive(svg);
    }
    
    // Reset zoom level for new SVG
    resetZoom();
}

// Make sure SVG has a proper viewBox attribute
function ensureViewBox(svg) {
    // If viewBox is missing but width/height present, add viewBox
    if (!svg.getAttribute('viewBox') && 
        svg.getAttribute('width') && 
        svg.getAttribute('height')) {
        
        const width = parseFloat(svg.getAttribute('width'));
        const height = parseFloat(svg.getAttribute('height'));
        
        svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    }
    
    // Make sure SVG has preserveAspectRatio attribute
    if (!svg.getAttribute('preserveAspectRatio')) {
        svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    }
}

// Handle window resize
function handleResize() {
    if (currentSvg) {
        // Re-process current SVG on resize
        processSvg(currentSvg);
    }
}

// Add interactive capabilities to fullscreen SVG
function makeFullscreenSvgInteractive(svg) {
    let isPanning = false;
    let startX, startY;
    let translateX = 0;
    let translateY = 0;
    
    // Set initial transform origin
    svg.style.transformOrigin = 'center center';
    
    // Mouse down event to start panning
    svg.addEventListener('mousedown', (e) => {
        // Only activate on left mouse button
        if (e.button !== 0) return;
        
        isPanning = true;
        startX = e.clientX;
        startY = e.clientY;
        svg.style.cursor = 'grabbing';
    });
    
    // Mouse move for panning
    window.addEventListener('mousemove', (e) => {
        if (!isPanning) return;
        
        const dx = e.clientX - startX;
        const dy = e.clientY - startY;
        
        translateX += dx;
        translateY += dy;
        
        startX = e.clientX;
        startY = e.clientY;
        
        svg.style.transform = `translate(${translateX}px, ${translateY}px) scale(${zoomLevel})`;
    });
    
    // Mouse up to stop panning
    window.addEventListener('mouseup', () => {
        isPanning = false;
        svg.style.cursor = 'grab';
    });
    
    // Mouse leave to stop panning
    svg.addEventListener('mouseleave', () => {
        isPanning = false;
        svg.style.cursor = 'grab';
    });
    
    // Set initial cursor
    svg.style.cursor = 'grab';
}
