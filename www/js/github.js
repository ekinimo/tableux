/**
 * GitHub link and fullscreen functionality
 */

export function setupGitHubLink() {
    const githubLink = document.getElementById('github-link');
    if (githubLink) {
        // Set the actual GitHub repository URL here
        const repoUrl = 'https://github.com/yourusername/analytical-tableaux-visualizer';
        
        githubLink.addEventListener('click', (e) => {
            e.preventDefault();
            window.open(repoUrl, '_blank');
        });
    }
}

export function setupFullscreenButton() {
    const fullscreenBtn = document.getElementById('fullscreen-btn');
    const visualizationContainer = document.querySelector('.visualization-container');
    
    if (!fullscreenBtn || !visualizationContainer) return;
    
    let isFullscreen = false;
    
    fullscreenBtn.addEventListener('click', () => {
        if (!isFullscreen) {
            // Enter fullscreen mode
            visualizationContainer.classList.add('visualization-fullscreen');
            fullscreenBtn.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3"></path>
                </svg>
                Exit Fullscreen
            `;
            isFullscreen = true;
            
            // Add escape key handler for exiting fullscreen
            document.addEventListener('keydown', exitFullscreenOnEscape);
            
            // Add close button to fullscreen mode
            const closeBtn = document.createElement('button');
            closeBtn.className = 'fullscreen-btn';
            closeBtn.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
            `;
            closeBtn.addEventListener('click', exitFullscreen);
            closeBtn.id = 'fullscreen-close-btn';
            visualizationContainer.appendChild(closeBtn);
        } else {
            exitFullscreen();
        }
    });
    
    function exitFullscreenOnEscape(e) {
        if (e.key === 'Escape' && isFullscreen) {
            exitFullscreen();
        }
    }
    
    function exitFullscreen() {
        visualizationContainer.classList.remove('visualization-fullscreen');
        fullscreenBtn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path>
            </svg>
            Fullscreen
        `;
        isFullscreen = false;
        document.removeEventListener('keydown', exitFullscreenOnEscape);
        
        // Remove close button
        const closeBtn = document.getElementById('fullscreen-close-btn');
        if (closeBtn) {
            closeBtn.remove();
        }
    }
}
