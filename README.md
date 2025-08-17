# drl-ball-balance

## Pygame Renderer (Alternative to HTML/Canvas)

A native Pygame renderer is available to visualize and simulate the Ball Balance environment without the HTML/canvas client.

### Install

```bash
pip install -r requirements.txt
```

### Run

1) Start the backend server:

```bash
python backend/main.py
```

2) In a separate terminal, start the Pygame renderer:

```bash
python renderers/pygame_renderer.py
```

Optional: Start your DRL agent so that actions are sent to the renderer via `ws://localhost:8000/ws/agent`. The renderer connects to `ws://localhost:8000/ws` and will receive `direction` and `reset` signals from the server.

Controls in the Pygame window:
- Press `R` to locally reset the visualization state
- Press `ESC` or close the window to quit

Notes:
- The HTML client in `public/` remains available but is no longer required when using the Pygame renderer.
- Physics, state updates, and message formats match the existing frontend protocol (`ballX`, `platformAngle`, `platformVelocity`, `direction`, `reset`).
