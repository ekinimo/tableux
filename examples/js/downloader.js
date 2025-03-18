/**
 * Downloader module - Handles SVG downloading functionality
 */

/**
 * Set up SVG download functionality
 * @param {Object} visualizer - The visualizer instance
 */
export function setupDownloader(visualizer) {
    const downloadButton = document.getElementById('download-svg');
    
    if (!downloadButton) {
        console.error("Download button not found");
        return;
    }
    
    downloadButton.addEventListener('click', () => {
        downloadCurrentSvg(visualizer);
    });
}

/**
 * Download the current SVG visualization
 * @param {Object} visualizer - The visualizer instance
 */
function downloadCurrentSvg(visualizer) {
    try {
        // Get current SVG content
        const svgString = visualizer.generateSvg(visualizer);
        
        // Check if there's content to download
        if (!svgString || svgString.trim() === '') {
            alert('No visualization to download. Please generate a tableaux first.');
            return;
        }
        
        // Get the SVG and fix any attributes for download
        const parser = new DOMParser();
        const svgDoc = parser.parseFromString(svgString, 'image/svg+xml');
        const svg = svgDoc.documentElement;
        
        // Ensure SVG has a proper viewBox
        if (!svg.hasAttribute('viewBox') && 
            svg.hasAttribute('width') && 
            svg.hasAttribute('height')) {
            
            const width = svg.getAttribute('width');
            const height = svg.getAttribute('height');
            
            svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
        }
        
        // Reset width/height to original values for download
        if (svg.dataset && svg.dataset.originalWidth) {
            svg.setAttribute('width', svg.dataset.originalWidth);
        }
        if (svg.dataset && svg.dataset.originalHeight) {
            svg.setAttribute('height', svg.dataset.originalHeight);
        }
        
        // Ensure background color is included
        const bgColor = document.body.classList.contains('dark-theme') ? '#000000' : '#ffffff';
        svg.setAttribute('style', `background-color: ${bgColor};`);
        
        // Serialize back to string
        const serializer = new XMLSerializer();
        const fixedSvgString = serializer.serializeToString(svg);
        
        // Create a blob from the SVG string
        const blob = new Blob([fixedSvgString], { type: 'image/svg+xml' });
        const url = URL.createObjectURL(blob);
        
        // Create a temporary anchor to trigger download
        const downloadLink = document.createElement('a');
        
        // Generate a filename based on the current formula
        const formula = document.getElementById('formula')?.value || '';
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').replace('T', '_').slice(0, -5);
        const filename = formula
            ? `tableaux_${formula.substring(0, 20).replace(/[^a-z0-9]/gi, '_')}_${timestamp}.svg`
            : `tableaux_${timestamp}.svg`;
        
        // Set link properties
        downloadLink.href = url;
        downloadLink.download = filename;
        
        // Add to DOM, click, and remove
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);
        
        // Clean up the URL object
        setTimeout(() => {
            URL.revokeObjectURL(url);
        }, 100);
        
    } catch (error) {
        console.error('Error downloading SVG:', error);
        alert('Failed to download SVG: ' + error.message);
    }
}

/**
 * Convert SVG to PNG and download it
 * Note: This is an optional feature that can be exposed via UI if needed
 * @param {Object} visualizer - The visualizer instance 
 * @param {number} scale - Scale factor for PNG conversion (default: 2)
 */
export function downloadAsPng(visualizer, scale = 2) {
    try {
        // Get current SVG content
        const svgString = visualizer.generateSvg(visualizer);
        
        // Check if there's content to download
        if (!svgString || svgString.trim() === '') {
            alert('No visualization to download. Please generate a tableaux first.');
            return;
        }
        
        // Create a container for the SVG
        const container = document.createElement('div');
        container.innerHTML = svgString;
        const svgElement = container.firstChild;
        
        // Get SVG dimensions
        const svgWidth = parseInt(svgElement.getAttribute('width') || '800');
        const svgHeight = parseInt(svgElement.getAttribute('height') || '600');
        
        // Create a canvas
        const canvas = document.createElement('canvas');
        canvas.width = svgWidth * scale;
        canvas.height = svgHeight * scale;
        const ctx = canvas.getContext('2d');
        
        // Set background color
        ctx.fillStyle = document.body.classList.contains('dark-theme') ? '#000000' : '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Create an image from the SVG
        const img = new Image();
        const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
        const url = URL.createObjectURL(svgBlob);
        
        img.onload = function() {
            // Draw the image to the canvas, scaled
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            
            // Convert to PNG and download
            const pngUrl = canvas.toDataURL('image/png');
            
            // Generate filename
            const formula = document.getElementById('formula')?.value || '';
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').replace('T', '_').slice(0, -5);
            const filename = formula
                ? `tableaux_${formula.substring(0, 20).replace(/[^a-z0-9]/gi, '_')}_${timestamp}.png`
                : `tableaux_${timestamp}.png`;
            
            // Create download link
            const downloadLink = document.createElement('a');
            downloadLink.href = pngUrl;
            downloadLink.download = filename;
            
            // Trigger download
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
            
            // Clean up
            URL.revokeObjectURL(url);
        };
        
        img.src = url;
        
    } catch (error) {
        console.error('Error downloading PNG:', error);
        alert('Failed to download PNG: ' + error.message);
    }
}
