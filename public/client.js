const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let state = {
  ball: { x: 0, v: 0 },
  platform: { angle: 0, angularVelocity: 0 },
  controlInput: 0
};

const GRAVITY = 6.81;

// time step use to calculate physics
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

  // Heads-up display (HUD) with key metrics
  const normalizedBallX = state.ball.x; // [-1, 1]
  const ballDistance = Math.abs(normalizedBallX);
  const action = Number(state.controlInput) || 0;
  const angleDeg = (state.platform.angle * 180) / Math.PI;
  const angularVelocity = state.platform.angularVelocity;
  // DRL reward currently: 1 - |ballX|
  const reward = 1 - Math.min(1, ballDistance);

  // Draw translucent background for readability
  ctx.save();
  ctx.font = "14px monospace";
  const lines = [
    `Ball X: ${normalizedBallX.toFixed(3)} (|x|=${ballDistance.toFixed(3)})`,
    `Angle: ${angleDeg.toFixed(1)}°`,
    `Ang Vel: ${angularVelocity.toFixed(3)}`,
    `Action: ${action.toFixed(3)}`,
    `Reward: ${reward.toFixed(3)}`
  ];
  const padding = 10;
  const lineHeight = 18;
  const boxWidth = 260;
  const boxHeight = padding * 2 + lines.length * lineHeight;
  ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
  ctx.fillRect(10, 10, boxWidth, boxHeight);
  ctx.fillStyle = "#fff";
  lines.forEach((text, i) => {
    ctx.fillText(text, 20, 10 + padding + (i + 1) * lineHeight - 4);
  });
  ctx.restore();
  requestAnimationFrame(draw);
}

function loop() {
  updatePhysics();
  requestAnimationFrame(loop);
}

loop();
draw();
