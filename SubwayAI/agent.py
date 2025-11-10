import json, time, random, os, csv, subprocess, numpy as np
from PIL import Image
import io
import argparse
import threading
import sys

ACTIONS          = ['left','right','jump','down','none']
GRID_W, GRID_H   = 5, 3
STATE_DIM        = GRID_W * GRID_H
MEMORY_SIZE      = 100_000
BATCH            = 32
GAMMA            = 0.99
EPS_START        = 1.0
EPS_END          = 0.05
EPS_DECAY        = 500_000
LR               = 1e-3
REPLACE_TARGET_EVERY = 1_000
CHECKPOINT_EVERY = 5_000

class NN:
    def __init__(self, layers):
        self.shapes = [(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        self.weights = [np.random.randn(*s) * 0.1 for s in self.shapes]
        self.biases  = [np.zeros((1, s[1])) for s in self.shapes]

    def forward(self, x):
        for w,b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.maximum(x @ w + b, 0)
        return x @ self.weights[-1] + self.biases[-1]

    def copy_from(self, other):
        self.weights = [w.copy() for w in other.weights]
        self.biases  = [b.copy() for b in other.biases]

    def save(self, path):
        np.savez(path, *self.weights, *self.biases)

    def load(self, path):
        data = np.load(path)
        n = len(self.weights)
        self.weights = [data[f'arr_{i}'] for i in range(n)]
        self.biases  = [data[f'arr_{i}'] for i in range(n, 2*n)]

online_net  = NN([STATE_DIM, 64, 32, len(ACTIONS)])
target_net  = NN([STATE_DIM, 64, 32, len(ACTIONS)])
target_net.copy_from(online_net)

memory, idx = [None]*MEMORY_SIZE, 0
step, episode = 0, 0

def push(s,a,r,s_,d):
    global idx
    memory[idx] = (s,a,r,s_,d)
    idx = (idx+1)%MEMORY_SIZE

def sample():
    batch = random.sample([m for m in memory if m is not None], BATCH)
    ss,aa,rr,ss_,dd = zip(*batch)
    return np.array(ss), aa, rr, np.array(ss_), dd

def train_step():
    if idx < BATCH: return
    
    try:
        s, a, r, s_, d = sample()
        
        # Ensure proper shapes
        s = np.array(s, dtype=np.float32)
        s_ = np.array(s_, dtype=np.float32)
        a = np.array(a, dtype=np.int32)
        r = np.array(r, dtype=np.float32)
        d = np.array(d, dtype=np.float32)
        
        # Forward pass through target network for next state values
        q_next = target_net.forward(s_)
        q_next_max = np.max(q_next, axis=1)
        y = r + GAMMA * q_next_max * (1 - d)
        
        # Forward pass through online network
        q_values = online_net.forward(s)
        q_selected = q_values[np.arange(BATCH), a]
        
        # Calculate TD error
        td_error = q_selected - y
        
        # Clip TD error to prevent exploding gradients
        td_error = np.clip(td_error, -1.0, 1.0)
        
        # Simple gradient approximation for stability
        learning_rate = LR * 0.01  # Very conservative learning rate
        
        # Update only the output layer weights for stability
        if len(online_net.weights) > 0:
            # Get the last hidden layer activations
            hidden_activations = s
            for w, b in zip(online_net.weights[:-1], online_net.biases[:-1]):
                hidden_activations = np.maximum(hidden_activations @ w + b, 0)
            
            # Update output layer
            output_w = online_net.weights[-1]
            output_b = online_net.biases[-1]
            
            # Compute gradients with proper shapes
            grad_w = np.zeros_like(output_w)
            grad_b = np.zeros_like(output_b)
            
            for i in range(BATCH):
                action_idx = a[i]
                error = td_error[i]
                
                # Update weights for the selected action
                grad_w[:, action_idx] += hidden_activations[i] * error * learning_rate
                grad_b[0, action_idx] += error * learning_rate
            
            # Apply gradients with clipping
            grad_w = np.clip(grad_w, -0.1, 0.1)
            grad_b = np.clip(grad_b, -0.1, 0.1)
            
            online_net.weights[-1] -= grad_w
            online_net.biases[-1] -= grad_b
            
    except Exception as e:
        print(f"⚠️  Training error: {e}")
        # Continue without updating to maintain stability

# BlueStacks and ADB configuration
BLUESTACKS_ADB_PORT = 5556
SUBWAY_SURFERS_PACKAGE = "com.kiloo.subwaysurf"
SCREEN_WIDTH = 720
SCREEN_HEIGHT = 1280

# Game state variables
prev_state = np.zeros(STATE_DIM)
prev_action = 0
prev_reward = 0
game_running = False

# Performance monitoring variables
current_fps = 0
frame_times = []
last_frame_time = time.time()
train_position = 2  # Center position (0=left, 1=center-left, 2=center, 3=center-right, 4=right)

def find_adb_path():
    """Find ADB executable path on Windows"""
    # Common ADB installation paths on Windows
    possible_paths = [
        r"C:\adb\platform-tools\adb.exe",
        r"C:\Android\platform-tools\adb.exe", 
        r"C:\Users\%USERNAME%\AppData\Local\Android\Sdk\platform-tools\adb.exe",
        r"C:\Program Files\Android\Android Studio\platform-tools\adb.exe",
        "adb"  # Try system PATH as fallback
    ]
    
    for path in possible_paths:
        try:
            # Expand environment variables
            expanded_path = os.path.expandvars(path)
            
            # Test if ADB works at this path
            if path == "adb":
                # Test system PATH
                result = subprocess.run("adb version", shell=True, capture_output=True, text=True, timeout=5)
            else:
                # Test specific path
                if os.path.exists(expanded_path):
                    result = subprocess.run(f'"{expanded_path}" version', shell=True, capture_output=True, text=True, timeout=5)
                else:
                    continue
            
            if result.returncode == 0:
                return path if path == "adb" else expanded_path
        except:
            continue
    
    return None

# Find ADB path at startup
ADB_PATH = find_adb_path()

def run_adb_command(command):
    """Execute ADB command and return output"""
    global ADB_PATH
    
    if ADB_PATH is None:
        print("❌ ADB not found. Please install ADB and add it to PATH, or install Android SDK.")
        print("💡 Download ADB: https://developer.android.com/studio/command-line/adb")
        return None
    
    try:
        # Use full path to ADB if not in system PATH
        if ADB_PATH == "adb":
            adb_command = f"adb {command}"
        else:
            adb_command = f'"{ADB_PATH}" {command}'
            
        result = subprocess.run(adb_command, shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print(f"⚠️  ADB command failed: {command}")
            print(f"Error: {result.stderr}")
            return None
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print(f"⚠️  ADB command timed out: {command}")
        return None
    except Exception as e:
        print(f"⚠️  ADB command error: {e}")
        return None

def check_bluestacks_running():
    """Check if BlueStacks is running and ADB is connected"""
    try:
        output = run_adb_command("devices")
        if output and "127.0.0.1:5556" in output and "device" in output:
            return True
        return False
    except:
        return False

def connect_to_bluestacks():
    """Connect to BlueStacks via ADB with comprehensive retry logic"""
    print("🔗 Connecting to BlueStacks...")
    print("📍 BlueStacks path: C:\\Program Files\\BlueStacks_nxt")
    
    max_retries = 5
    for attempt in range(max_retries):
        if attempt > 0:
            print(f"🔄 Retry attempt {attempt + 1}/{max_retries}")
        
        # First, try to kill any existing ADB server
        if attempt == 0:
            print("🔧 Initializing ADB server...")
            run_adb_command("kill-server")
            time.sleep(1)
            run_adb_command("start-server")
            time.sleep(2)
        
        # Try to connect to BlueStacks ADB port
        print(f"🔌 Attempting connection to 127.0.0.1:{BLUESTACKS_ADB_PORT}...")
        result = run_adb_command(f"connect 127.0.0.1:{BLUESTACKS_ADB_PORT}")
        
        if result is None:
            if attempt < max_retries - 1:
                print("⏳ ADB command failed, waiting 5 seconds before retry...")
                time.sleep(5)
                continue
            else:
                print("❌ Failed to execute ADB connect command")
                print("💡 Troubleshooting:")
                print("   • Install ADB: https://developer.android.com/studio/command-line/adb")
                print("   • Add ADB to system PATH")
                print("   • Restart command prompt after installation")
                return False
        
        # Wait for connection to establish
        time.sleep(3)
        
        # Verify connection with detailed feedback
        print("🔍 Verifying connection...")
        if check_bluestacks_running():
            print("✅ BlueStacks connected successfully!")
            print("🎯 ADB connection established")
            return True
        else:
            if attempt < max_retries - 1:
                print(f"⏳ Connection not established, retrying in 5 seconds... ({attempt + 1}/{max_retries})")
                time.sleep(5)
            else:
                print("❌ Failed to connect to BlueStacks after all retries")
                print("\n🔧 Detailed Troubleshooting Guide:")
                print("   1. ✅ Ensure BlueStacks is running from: C:\\Program Files\\BlueStacks_nxt")
                print("   2. ✅ Enable ADB debugging:")
                print("      • Open BlueStacks Settings")
                print("      • Go to Advanced → Developer Options")
                print("      • Enable 'USB Debugging' or 'ADB Debugging'")
                print("   3. ✅ Check BlueStacks ADB port:")
                print("      • BlueStacks X uses port 5556")
                print("      • Restart BlueStacks if needed")
                print("   4. ✅ Firewall/Antivirus:")
                print("      • Allow ADB and BlueStacks through firewall")
                print("   5. ✅ Try manual connection:")
                print("      • Run: adb connect 127.0.0.1:5556")
                print("      • Run: adb devices")
                return False
    
    return False

def is_app_running():
    """Check if Subway Surfers app is currently running"""
    output = run_adb_command(f"shell pidof {SUBWAY_SURFERS_PACKAGE}")
    return output is not None and output.strip() != ""

def launch_subway_surfers():
    """Launch Subway Surfers app on BlueStacks"""
    print("🚇 Launching Subway Surfers...")
    
    if is_app_running():
        print("ℹ️  Subway Surfers is already running")
        return True
    
    # Launch the app
    result = run_adb_command(f"shell monkey -p {SUBWAY_SURFERS_PACKAGE} -c android.intent.category.LAUNCHER 1")
    if result is None:
        print("❌ Failed to launch Subway Surfers")
        return False
    
    # Wait for app to start
    print("⏳ Waiting for app to start...")
    for i in range(10):
        time.sleep(1)
        if is_app_running():
            print("✅ Subway Surfers launched successfully")
            return True
    
    print("❌ Subway Surfers failed to start within 10 seconds")
    return False

def restart_game():
    """Restart the Subway Surfers game for a new episode"""
    print("🔄 Restarting game for new episode...")
    
    # Force stop the app
    run_adb_command(f"shell am force-stop {SUBWAY_SURFERS_PACKAGE}")
    time.sleep(1)
    
    # Launch it again
    return launch_subway_surfers()

def capture_screen():
    """Capture screenshot from BlueStacks via ADB"""
    try:
        # Capture screenshot using ADB
        result = subprocess.run("adb exec-out screencap -p", shell=True, capture_output=True, timeout=5)
        if result.returncode != 0:
            return None
        
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(result.stdout))
        return image
    except Exception as e:
        print(f"⚠️  Screen capture failed: {e}")
        return None

# Touch input timing and debouncing
last_action_time = 0
last_action = None
MIN_ACTION_INTERVAL = 0.15  # Minimum time between actions

def send_touch(action):
    """Send touch gesture to BlueStacks based on action with timing control"""
    global last_action_time, last_action
    
    current_time = time.time()
    
    # Debouncing: prevent rapid repeated actions
    if (current_time - last_action_time < MIN_ACTION_INTERVAL and 
        action == last_action and action != 'none'):
        return
    
    # Map actions to appropriate touch coordinates and gestures
    if action == 'left':
        # Swipe left - from center to left
        start_x, start_y = SCREEN_WIDTH//2, SCREEN_HEIGHT//2
        end_x, end_y = SCREEN_WIDTH//4, SCREEN_HEIGHT//2
        duration = 120
        run_adb_command(f"shell input swipe {start_x} {start_y} {end_x} {end_y} {duration}")
        
    elif action == 'right':
        # Swipe right - from center to right
        start_x, start_y = SCREEN_WIDTH//2, SCREEN_HEIGHT//2
        end_x, end_y = 3*SCREEN_WIDTH//4, SCREEN_HEIGHT//2
        duration = 120
        run_adb_command(f"shell input swipe {start_x} {start_y} {end_x} {end_y} {duration}")
        
    elif action == 'jump':
        # Swipe up - from center upward
        start_x, start_y = SCREEN_WIDTH//2, SCREEN_HEIGHT//2
        end_x, end_y = SCREEN_WIDTH//2, SCREEN_HEIGHT//4
        duration = 150
        run_adb_command(f"shell input swipe {start_x} {start_y} {end_x} {end_y} {duration}")
        
    elif action == 'down':
        # Swipe down - from center downward
        start_x, start_y = SCREEN_WIDTH//2, SCREEN_HEIGHT//2
        end_x, end_y = SCREEN_WIDTH//2, 3*SCREEN_HEIGHT//4
        duration = 100
        run_adb_command(f"shell input swipe {start_x} {start_y} {end_x} {end_y} {duration}")
    
    # 'none' action does nothing
    
    # Update timing tracking
    if action != 'none':
        last_action_time = current_time
        last_action = action

def detect_game_over(image):
    """Detect if the game is over by analyzing the screenshot"""
    if image is None:
        return False
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Look for game over indicators
    # Check for common game over screen patterns
    height, width = img_array.shape[:2]
    
    # Sample multiple areas for game over detection
    center_y = height // 2
    center_x = width // 2
    
    # Check center area for restart button
    sample_area = img_array[center_y-100:center_y+100, center_x-150:center_x+150]
    
    # Check upper area for "Game Over" text
    upper_area = img_array[height//3:height//2, center_x-200:center_x+200]
    
    # Look for bright colors that might indicate game over screen
    for area in [sample_area, upper_area]:
        if area.size > 0:
            # Check for high contrast areas (buttons, text)
            gray = np.mean(area, axis=2) if len(area.shape) == 3 else area
            contrast = np.std(gray)
            
            # If there's high contrast in center (likely UI elements), might be game over
            if contrast > 60:
                # Additional check: look for restart button colors
                if len(area.shape) == 3:
                    # Look for green/blue button colors common in game over screens
                    green_pixels = np.sum((area[:,:,1] > 150) & (area[:,:,0] < 100) & (area[:,:,2] < 100))
                    blue_pixels = np.sum((area[:,:,2] > 150) & (area[:,:,0] < 100) & (area[:,:,1] < 100))
                    white_pixels = np.sum((area[:,:,0] > 200) & (area[:,:,1] > 200) & (area[:,:,2] > 200))
                    
                    if green_pixels > 50 or blue_pixels > 50 or white_pixels > 500:
                        return True
    
    # Check for pause screen or menu indicators
    # Look for consistent UI patterns
    if len(img_array.shape) == 3:
        # Check for dark overlay (common in game over screens)
        dark_overlay = np.sum((img_array[:,:,0] < 50) & (img_array[:,:,1] < 50) & (img_array[:,:,2] < 50))
        total_pixels = img_array.shape[0] * img_array.shape[1]
        
        if dark_overlay > total_pixels * 0.3:  # More than 30% dark pixels
            return True
    
    # Fallback: very small random chance for testing
    return random.random() < 0.001

def crop_game_area(image):
    """Crop the image to focus on the main game area"""
    if image is None:
        return None
    
    # Convert to numpy array
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Crop to focus on the main game area (remove UI elements)
    # Typical Subway Surfers layout: game area is in the center
    crop_top = int(height * 0.15)    # Remove top UI
    crop_bottom = int(height * 0.85)  # Remove bottom UI
    crop_left = int(width * 0.1)     # Remove side margins
    crop_right = int(width * 0.9)    # Remove side margins
    
    cropped = img_array[crop_top:crop_bottom, crop_left:crop_right]
    return cropped

def analyze_pixels(cropped_area):
    """Analyze cropped game area to detect objects with improved accuracy"""
    if cropped_area is None or cropped_area.size == 0:
        return np.zeros((GRID_H, GRID_W), dtype=int)
    
    height, width = cropped_area.shape[:2]
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    
    # Divide the cropped area into a 3x5 grid
    cell_height = height // GRID_H
    cell_width = width // GRID_W
    
    for row in range(GRID_H):
        for col in range(GRID_W):
            # Extract cell
            y1 = row * cell_height
            y2 = min((row + 1) * cell_height, height)
            x1 = col * cell_width
            x2 = min((col + 1) * cell_width, width)
            
            cell = cropped_area[y1:y2, x1:x2]
            
            if cell.size == 0:
                continue
            
            # Analyze cell content
            if len(cell.shape) == 3:  # Color image
                # Calculate average colors and color variance
                avg_r = np.mean(cell[:,:,0])
                avg_g = np.mean(cell[:,:,1])
                avg_b = np.mean(cell[:,:,2])
                
                # Calculate color variance for better detection
                var_r = np.var(cell[:,:,0])
                var_g = np.var(cell[:,:,1])
                var_b = np.var(cell[:,:,2])
                
                # Improved classification based on Subway Surfers colors
                # Trains (dark metallic colors)
                if (avg_r < 100 and avg_g < 100 and avg_b < 100) and (var_r + var_g + var_b > 200):
                    grid[row, col] = 1  # Train/obstacle
                # Barriers (red/orange warning colors)
                elif (avg_r > 120 and avg_g < 80 and avg_b < 80) or (avg_r > 150 and avg_g > 100 and avg_b < 80):
                    grid[row, col] = 2  # Barrier
                # Coins (bright yellow/gold)
                elif (avg_r > 180 and avg_g > 180 and avg_b < 120) or (avg_r > 200 and avg_g > 150 and avg_b < 100):
                    grid[row, col] = 3  # Coin
                # Power-ups and ramps (blue/purple/green)
                elif (avg_b > 120 and avg_r < 100) or (avg_g > 120 and avg_r < 100 and avg_b < 100):
                    grid[row, col] = 4  # Power-up/ramp
                # Character area (skin tones, clothing)
                elif avg_r > 100 and avg_g > 80 and avg_b > 60 and abs(avg_r - avg_g) < 50:
                    grid[row, col] = 0  # Character/background (treat as empty for navigation)
                # Very dark areas (shadows, deep obstacles)
                elif avg_r < 50 and avg_g < 50 and avg_b < 50:
                    grid[row, col] = 1  # Dark obstacle
                else:
                    grid[row, col] = 0  # Background/empty
            else:  # Grayscale
                avg_intensity = np.mean(cell)
                intensity_var = np.var(cell)
                
                if avg_intensity < 60:
                    grid[row, col] = 1  # Dark obstacle
                elif avg_intensity > 200 and intensity_var > 100:
                    grid[row, col] = 3  # Bright object (likely coin)
                else:
                    grid[row, col] = 0  # Background
    
    return grid

def analyze_screen(image):
    """Analyze screenshot and extract game state"""
    if image is None:
        return np.zeros(STATE_DIM)
    
    # Crop to game area
    cropped = crop_game_area(image)
    
    # Analyze pixels to get grid
    grid = analyze_pixels(cropped)
    
    # Flatten grid to state vector
    state = grid.flatten().astype(np.float32)
    
    return state

def wait_for_game_start():
    """Wait for the game to be ready and handle initial screens"""
    print("⏳ Waiting for game to be ready...")
    time.sleep(3)
    
    # Try to tap the screen to start the game or dismiss any popups
    for i in range(5):
        # Tap center of screen to start game or dismiss popups
        run_adb_command(f"shell input tap {SCREEN_WIDTH//2} {SCREEN_HEIGHT//2}")
        time.sleep(1)
        
        # Try tapping play button area (common location)
        run_adb_command(f"shell input tap {SCREEN_WIDTH//2} {2*SCREEN_HEIGHT//3}")
        time.sleep(1)
    
    print("✅ Game should be ready now")

def initialize_bluestacks():
    """Initialize BlueStacks connection and launch game"""
    print("🚀 Initializing BlueStacks AI Agent...")
    
    # Check ADB installation
    if ADB_PATH is None:
        print("❌ ADB not found!")
        print("💡 Installation Guide:")
        print("   1. Download ADB: https://developer.android.com/studio/command-line/adb")
        print("   2. Extract to C:\\adb\\platform-tools\\")
        print("   3. Add C:\\adb\\platform-tools to your system PATH")
        print("   4. Restart your command prompt")
        print("   5. Test with: adb version")
        return False
    else:
        print(f"✅ ADB found at: {ADB_PATH}")
    
    # Check if BlueStacks is running
    if not check_bluestacks_running():
        print("🔗 BlueStacks not detected, attempting to connect...")
        if not connect_to_bluestacks():
            print("❌ Failed to connect to BlueStacks")
            print("💡 Make sure BlueStacks is running and ADB debugging is enabled")
            return False
    else:
        print("✅ BlueStacks already connected")
    
    # Launch Subway Surfers
    if not launch_subway_surfers():
        print("❌ Failed to launch Subway Surfers")
        print("💡 Make sure Subway Surfers is installed on BlueStacks")
        return False
    
    # Wait for game to be ready
    wait_for_game_start()
    
    return True

def get_epsilon():
    """Get current exploration rate"""
    return max(EPS_END, EPS_START - (step / EPS_DECAY))

def choose_action(state):
    """Choose action using epsilon-greedy policy"""
    if random.random() < get_epsilon():
        return random.randint(0, len(ACTIONS)-1)
    else:
        q_values = online_net.forward(state.reshape(1, -1))
        return np.argmax(q_values)

def calculate_reward(prev_state, action, current_state, game_over):
    """Calculate reward based on game state"""
    if game_over:
        return -10  # Penalty for dying
    
    # Basic survival reward
    reward = 1
    
    # Additional rewards can be added based on coins collected, distance, etc.
    # This is a placeholder implementation
    
    return reward

def update_fps():
    """Update FPS calculation"""
    global current_fps, frame_times, last_frame_time
    
    current_time = time.time()
    frame_times.append(current_time - last_frame_time)
    last_frame_time = current_time
    
    # Keep only last 30 frames for FPS calculation
    if len(frame_times) > 30:
        frame_times.pop(0)
    
    if len(frame_times) > 1:
        avg_frame_time = sum(frame_times) / len(frame_times)
        current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

def get_train_ascii(position):
    """Get ASCII train representation based on position"""
    train_chars = ["🚂", "🚃", "🚃", "🚃", "🚂"]
    positions = ["     ", "  ", "", "  ", "     "]
    
    # Create train visualization
    train_line = ""
    for i in range(5):
        if i == position:
            train_line += "🏃"  # Runner character
        else:
            train_line += "░"
    
    return train_line

def print_dashboard(episode, steps, reward, epsilon, action="none"):
    """Print ASCII dashboard with current stats"""
    global train_position
    
    # Update train position based on action
    if action == "left" and train_position > 0:
        train_position -= 1
    elif action == "right" and train_position < 4:
        train_position += 1
    
    # Clear screen (Windows compatible)
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    print("┃                        🚇  S U B W A Y   A I  🏄‍♂️                   ┃")
    print("┃                \"Jake's autopilot on digital steroids\"                ┃")
    print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    print()
    
    # Game visualization
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  🎮  LIVE  GAME  VIEW                                                ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print("        ┌──────────────────────────────┐")
    print("        │  🚂   🚂   🚂   🚂   🚂    │   Trains ahead")
    print(f"        │  {get_train_ascii(train_position)}  │   AI Position")
    print("        │  🪙   🚧   🪙   🚧   🪙    │   Coins & Barriers")
    print("        └──────────────────────────────┘")
    print()
    
    # Stats dashboard
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  📊  PERFORMANCE  DASHBOARD                                          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"┌────────────┬────────────┬────────────┬────────────┬────────────┐")
    print(f"│ Episode    │ Steps      │ Reward     │ Epsilon    │ FPS        │")
    print(f"├────────────┼────────────┼────────────┼────────────┼────────────┤")
    print(f"│ {episode:10d} │ {steps:10d} │ {reward:10.1f} │ {epsilon:10.3f} │ {current_fps:10.1f} │")
    print(f"└────────────┴────────────┴────────────┴────────────┴────────────┘")
    print()
    
    # Action indicator
    action_arrows = {
        'left': '←', 'right': '→', 'jump': '↑', 'down': '↓', 'none': '•'
    }
    print(f"🎯 Current Action: {action_arrows.get(action, '•')} {action.upper()}")
    print()
    
    # Progress bar for epsilon decay
    epsilon_progress = int((1 - epsilon) * 50)  # 50 char progress bar
    progress_bar = "█" * epsilon_progress + "░" * (50 - epsilon_progress)
    print(f"🧠 Learning Progress: [{progress_bar}] {(1-epsilon)*100:.1f}%")
    print()
    
    # Instructions
    print("💡 Press Ctrl+C to stop training and save model")
    print("🔧 BlueStacks path: C:\\Program Files\\BlueStacks_nxt")

def save_run_data(episode, steps_survived, total_reward):
    """Save episode data to CSV file"""
    print(f"💾 Attempting to save episode {episode} data...")
    try:
        # Get script directory for relative paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        runs_file = os.path.join(script_dir, 'runs.csv')
        
        print(f"📁 CSV file path: {runs_file}")
        
        file_exists = os.path.exists(runs_file)
        print(f"📄 File exists: {file_exists}")
        
        # If file exists but has wrong headers, recreate it
        if file_exists:
            try:
                with open(runs_file, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line != 'episode,steps,reward,epsilon':
                        print("🔧 Fixing CSV headers...")
                        file_exists = False  # Force recreation with correct headers
            except:
                file_exists = False
        
        with open(runs_file, 'a' if file_exists else 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                print("📝 Writing CSV headers...")
                writer.writerow(['episode', 'steps', 'reward', 'epsilon'])
            
            epsilon_value = get_epsilon()
            print(f"📊 Writing data: episode={episode}, steps={steps_survived}, reward={total_reward:.1f}, epsilon={epsilon_value:.3f}")
            writer.writerow([episode, steps_survived, total_reward, epsilon_value])
        
        print(f"✅ Episode {episode} data saved successfully!")
        
    except Exception as e:
        print(f"❌ Failed to save run data: {e}")
        import traceback
        traceback.print_exc()

def main_game_loop():
    """Main game loop for training the AI agent"""
    global step, episode, prev_state, prev_action, prev_reward, game_running
    
    if not initialize_bluestacks():
        return
    
    print("🧠 Neural network loaded")
    print("🎮 Starting AI training...")
    
    # Load existing model if available
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(script_dir, 'model.npz')
    
    if os.path.exists(model_file):
        try:
            online_net.load(model_file)
            target_net.copy_from(online_net)
            print("📁 Loaded existing model")
        except:
            print("⚠️  Failed to load model, starting fresh")
    
    while True:
        try:
            episode += 1
            episode_steps = 0
            episode_reward = 0
            game_over = False
            
            print(f"\n🎯 Episode {episode} | Steps: {step} | ε: {get_epsilon():.3f}")
            
            # Capture initial state
            screenshot = capture_screen()
            current_state = analyze_screen(screenshot)
            
            while not game_over:
                # Choose action
                action_idx = choose_action(current_state)
                action = ACTIONS[action_idx]
                
                # Execute action
                send_touch(action)
                time.sleep(0.1)  # Small delay between actions
                
                # Capture new state
                screenshot = capture_screen()
                next_state = analyze_screen(screenshot)
                game_over = detect_game_over(screenshot)
                
                # Calculate reward
                reward = calculate_reward(current_state, action_idx, next_state, game_over)
                
                # Store experience
                if step > 0:  # Skip first step
                    push(prev_state, prev_action, prev_reward, current_state, False)
                
                # Train network
                if step % 4 == 0:  # Train every 4 steps
                    train_step()
                
                # Update target network
                if step % REPLACE_TARGET_EVERY == 0:
                    target_net.copy_from(online_net)
                    print(f"🔄 Target network updated at step {step}")
                
                # Save checkpoint
                if step % CHECKPOINT_EVERY == 0:
                    online_net.save(model_file)
                    print(f"💾 Model saved at step {step}")
                
                # Update for next iteration
                prev_state = current_state.copy()
                prev_action = action_idx
                prev_reward = reward
                current_state = next_state
                
                step += 1
                episode_steps += 1
                episode_reward += reward
                
                # Update FPS and display dashboard
                update_fps()
                if episode_steps % 10 == 0:  # Update dashboard every 10 steps
                    print_dashboard(episode, episode_steps, episode_reward, get_epsilon(), action)
            
            # Episode ended
            print(f"💀 Game Over! Episode {episode} - Steps: {episode_steps}, Reward: {episode_reward:.1f}")
            
            # Store final experience
            push(prev_state, prev_action, prev_reward, current_state, True)
            
            # Save episode data
            save_run_data(episode, episode_steps, episode_reward)
            
            # Restart game for next episode
            time.sleep(2)
            restart_game()
            wait_for_game_start()
            
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted by user")
            print("💾 Saving model before exit...")
            try:
                online_net.save(model_file)
                print("✅ Model saved successfully")
            except Exception as save_error:
                print(f"❌ Failed to save model: {save_error}")
            break
        except Exception as e:
            print(f"⚠️  Error in game loop: {e}")
            print("🔄 Attempting to recover...")
            
            # Save model as precaution
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                model_file = os.path.join(script_dir, 'model.npz')
                online_net.save(model_file)
                print("💾 Model saved as precaution")
            except:
                print("⚠️  Could not save model")
            
            # Wait before retry
            print("⏳ Waiting 10 seconds before retry...")
            time.sleep(10)
            
            # Try to reconnect to BlueStacks
            if not check_bluestacks_running():
                print("🔗 BlueStacks connection lost, attempting to reconnect...")
                if connect_to_bluestacks():
                    print("✅ Reconnected successfully")
                    # Restart game after reconnection
                    restart_game()
                    wait_for_game_start()
                else:
                    print("❌ Failed to reconnect, exiting...")
                    break
            else:
                # BlueStacks is still connected, just restart the game
                print("🔄 Restarting game...")
                restart_game()
                wait_for_game_start()

def human_play_mode():
    """Allow human to play while the system observes"""
    print("👤 Human play mode - AI will observe your gameplay")
    print("🎮 Play the game manually, AI will learn from your actions")
    print("⌨️  Press Ctrl+C to exit")
    
    if not initialize_bluestacks():
        return
    
    try:
        while True:
            # Just capture and analyze screens for learning
            screenshot = capture_screen()
            if screenshot:
                state = analyze_screen(screenshot)
                game_over = detect_game_over(screenshot)
                
                if game_over:
                    print("💀 Game over detected - waiting for restart...")
                    time.sleep(3)
            
            time.sleep(0.5)  # Slower polling for human play
            
    except KeyboardInterrupt:
        print("\n👋 Exiting human play mode")

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description='Subway Surfers AI Agent')
    parser.add_argument('--headless', action='store_true', 
                       help='Run BlueStacks in headless mode (minimize window)')
    parser.add_argument('--human', action='store_true', 
                       help='Disable AI and allow manual play while observing')
    parser.add_argument('--speedrun', action='store_true',
                       help='Enable speedrun mode for 10x faster learning')
    args = parser.parse_args()
    
    print("🚇 Subway Surfers AI Agent")
    print("=" * 50)
    
    if args.human:
        human_play_mode()
        return
    
    if args.headless:
        print("🔇 Headless mode: BlueStacks will run minimized")
        # Try to minimize BlueStacks window
        try:
            subprocess.run("powershell -Command \"(Get-Process BlueStacks* | Where-Object {$_.MainWindowTitle -ne ''}) | ForEach-Object {$_.CloseMainWindow()}\"", 
                         shell=True, capture_output=True)
        except:
            print("⚠️  Could not minimize BlueStacks window")
    
    if args.speedrun:
        print("🏃‍♂️ Speedrun mode: 10x faster learning enabled")
        global LR, EPS_DECAY, CHECKPOINT_EVERY
        LR *= 10  # Increase learning rate
        EPS_DECAY //= 10  # Faster epsilon decay
        CHECKPOINT_EVERY //= 2  # More frequent saves
    
    main_game_loop()

if __name__ == "__main__":
    main()