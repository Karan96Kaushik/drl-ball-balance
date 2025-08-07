const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let state = {
  ball: { x: 0, v: 0 },
  platform: { angle: 0, angularVelocity: 0 },
  controlInput: 0
};

const GRAVITY = 7.81;
const TIME_STEP = 0.016;
const FRICTION = 0.8;
const CONTROL_GAIN = 0.003;

// Connect to backend WebSocket
const socket = new WebSocket("ws://localhost:8000/ws");

function reset() {
  state.ball.x = 0;
  state.ball.v = 0;
  state.platform.angle = 0;
  state.platform.angularVelocity = 0;
  state.controlInput = 0;
}

socket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.direction !== undefined) {
    state.controlInput = data.direction;
  }
  if (data.reset) {
    reset();
  }
};

// Main physics loop
function updatePhysics() {
  // Smooth input
  state.controlInput = parseFloat(state.controlInput) || 0;
  state.platform.angularVelocity += state.controlInput * CONTROL_GAIN;
  state.platform.angle += state.platform.angularVelocity;
  state.platform.angularVelocity *= FRICTION;

  const accel = GRAVITY * Math.sin(state.platform.angle);
  state.ball.v += accel * TIME_STEP;
  state.ball.x += state.ball.v * TIME_STEP;

  // Boundaries
  if (state.ball.x < -1) {
    state.ball.x = -1;
    state.ball.v *= -0.5;
  }
  if (state.ball.x > 1) {
    state.ball.x = 1;
    state.ball.v *= -0.5;
  }

  // Send current state to backend
  if (socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({
      ballX: state.ball.x,
      platformAngle: state.platform.angle,
      platformVelocity: state.platform.angularVelocity
    }));
  }
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const centerX = canvas.width / 2;
  const centerY = canvas.height / 2;
  const platformLength = 300;

  ctx.save();
  ctx.translate(centerX, centerY);
  ctx.rotate(state.platform.angle);

  ctx.fillStyle = "#555";
  ctx.fillRect(-platformLength / 2, -5, platformLength, 10);

  const ballX = state.ball.x * (platformLength / 2);
  ctx.beginPath();
  ctx.arc(ballX, -15, 10, 0, Math.PI * 2);
  ctx.fillStyle = "#0f0";
  ctx.fill();

  ctx.restore();
  requestAnimationFrame(draw);
}

function loop() {
  updatePhysics();
  requestAnimationFrame(loop);
}

loop();
draw();
