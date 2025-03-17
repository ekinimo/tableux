import { initUI } from './ui.js';
import { initVisualizer } from './visualizer.js';
import { setupDownloader } from './downloader.js';

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', async () => {
    try {
        // Show loading indicator
        const loadingElement = document.getElementById('loading');
        loadingElement.classList.remove('hidden');
        
        // Initialize main components
        const visualizer = await initVisualizer();
        
        // Check if visualizer was loaded successfully
        if (!visualizer) {
            showError("Failed to initialize visualizer. Please ensure WebAssembly is supported by your browser.");
            loadingElement.classList.add('hidden');
            return;
        }
        
        // Initialize UI with visualizer instance
        initUI(visualizer);
        
        // Setup SVG download functionality
        setupDownloader(visualizer);
        
        // Hide loading indicator
        loadingElement.classList.add('hidden');
        
        console.log("Application initialized successfully");
    } catch (error) {
        console.error("Initialization error:", error);
        showError(`Failed to initialize: ${error.message || "Unknown error loading WASM module"}`);
        document.getElementById('loading').classList.add('hidden');
    }
});

// Show error message
function showError(message) {
    const errorElement = document.getElementById('error');
    errorElement.textContent = message;
    errorElement.classList.remove('hidden');
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

function updateThemeIcon(isDark) {
    const themeIcon = document.getElementById('theme-icon');
    
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
