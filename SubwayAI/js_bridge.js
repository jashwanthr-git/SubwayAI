(() => {
    // Wait until iframe is ready
    const wait = setInterval(() => {
        const iframe = document.querySelector('iframe');
        if (!iframe) return;
        clearInterval(wait);
        const doc = iframe.contentDocument || iframe.contentWindow.document;
        const canvas = doc.querySelector('canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        const GRID_W = 5,
            GRID_H = 3;
        const cellW = Math.floor(canvas.width / 8);
        const cellH = Math.floor(canvas.height / 6);
        const offsetX = Math.floor(canvas.width * 0.4);
        const offsetY = Math.floor(canvas.height * 0.5);

        function scanCell(x, y) {
            const img = ctx.getImageData(x, y, cellW, cellH).data;
            let counts = [0, 0, 0, 0, 0]; // bg, train, barrier, coin, ramp
            for (let i = 0; i < img.length; i += 4) {
                const [r, g, b] = [img[i], img[i + 1], img[i + 2]];
                if (r < 50 && g < 50 && b < 50) counts[1]++; // dark = train
                else if (r > 200 && g < 100 && b < 100) counts[2]++; // red = barrier
                else if (r > 200 && g > 200 && b < 100) counts[3]++; // yellow = coin
                else if (Math.abs(r - g) < 20 && b > 200) counts[4]++; // blue = ramp
            }
            return counts.indexOf(Math.max(...counts));
        }

        function getState() {
            const state = new Array(GRID_W * GRID_H);
            for (let gx = 0; gx < GRID_W; gx++) {
                for (let gy = 0; gy < GRID_H; gy++) {
                    const x = offsetX + gx * cellW;
                    const y = offsetY + gy * cellH - cellH;
                    state[gx * GRID_H + gy] = scanCell(x, y);
                }
            }
            return state;
        }

        // WebSocket connection with retry logic
        let ws = null;
        let stateInterval = null;
        
        function connectWebSocket() {
            try {
                ws = new WebSocket('ws://localhost:8765');
                
                ws.onopen = () => {
                    console.log('🔗 Connected to AI agent');
                    // Start sending game state at 30 FPS
                    stateInterval = setInterval(() => {
                        if (ws && ws.readyState === WebSocket.OPEN) {
                            ws.send(JSON.stringify({ type: 'state', state: getState() }));
                        }
                    }, 33);
                };
                
                ws.onmessage = (evt) => {
                    const { action } = JSON.parse(evt.data);
                    const keyMap = { 
                        left: 'ArrowLeft', 
                        right: 'ArrowRight', 
                        jump: 'Space', 
                        down: 'ArrowDown', 
                        none: null 
                    };
                    const key = keyMap[action];
                    if (key) {
                        doc.dispatchEvent(new KeyboardEvent('keydown', { code: key }));
                        setTimeout(() => doc.dispatchEvent(new KeyboardEvent('keyup', { code: key })), 120);
                    }
                };
                
                ws.onclose = () => {
                    console.log('🔌 WebSocket disconnected, retrying...');
                    if (stateInterval) {
                        clearInterval(stateInterval);
                        stateInterval = null;
                    }
                    // Retry connection after 2 seconds
                    setTimeout(connectWebSocket, 2000);
                };
                
                ws.onerror = (error) => {
                    console.error('❌ WebSocket error:', error);
                };
                
            } catch (error) {
                console.error('❌ Failed to create WebSocket:', error);
                setTimeout(connectWebSocket, 2000);
            }
        }
        
        // Start WebSocket connection
        connectWebSocket();

        // Detect death by looking for white flash
        let alive = true;
        setInterval(() => {
            const img = ctx.getImageData(canvas.width / 2, canvas.height / 2, 5, 5).data;
            const white = img.filter(v => v > 240).length;
            if (white > 30 && alive) {
                alive = false;
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'dead' }));
                }
                setTimeout(() => {
                    alive = true;
                    location.reload();
                }, 3000);
            }
        }, 100);
    }, 500);
})();