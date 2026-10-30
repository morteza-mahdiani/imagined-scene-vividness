const CELLS = {
  0: {
    name: "Middle",
    also: "",
    color: "#5e5a54",
    text: "",
    prefix: "",
  },
  1: {
    name: "Agreement, high",
    also: "Shared-vivid",
    color: "#3c6e9a",
    text: "Humans and the shared component both place these images high. The examples are salient single subjects: an animal, a player, a child.",
    prefix: "agree-high",
  },
  2: {
    name: "Agreement, low",
    also: "Shared-dull",
    color: "#3c6e9a",
    text: "Humans and the shared component both place these images low. The examples are ordinary scenes rather than a single prominent subject.",
    prefix: "agree-low",
  },
  3: {
    name: "Underpredicted",
    also: "Machine-blind",
    color: "#c06a2c",
    text: "Humans rate these images high and the shared component does not. The examples are indoor and structured scenes.",
    prefix: "under",
  },
  4: {
    name: "Overpredicted",
    also: "False-vivid",
    color: "#2e6b5e",
    text: "The shared component is high and human vividness is low. The examples are visually salient single subjects.",
    prefix: "over",
  },
};

const canvas = document.querySelector("#scatter");
const ctx = canvas.getContext("2d");
const titleEl = document.querySelector("#cell-title");
const copyEl = document.querySelector("#cell-copy");
const gallery = document.querySelector("#gallery");
const tip = document.querySelector("#tip");
const tipImg = tip.querySelector("img");
const tipName = tip.querySelector("strong");
const tipMeta = tip.querySelector("span");
const chips = [...document.querySelectorAll(".chips button")];

let data = null;
let sx = [];
let sy = [];
let hover = -1;
let selected = 1;
let layout = null;

function imagesFor(cell) {
  const prefix = CELLS[cell].prefix;
  return Array.from({ length: 6 }, (_, i) => `images/${prefix}-${i}.jpg`);
}

function showCell(cell) {
  selected = cell;
  const info = CELLS[cell];
  titleEl.textContent = info.name;
  titleEl.style.color = info.color;
  copyEl.textContent = `${info.text} In the paper this cell is also called ${info.also.toLowerCase()}.`;
  gallery.innerHTML = imagesFor(cell).map((src) => `<img src="${src}" alt="">`).join("");
  chips.forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.cell) === cell)));
  draw();
}

function resize() {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  layout = {
    dpr,
    w: canvas.width,
    h: canvas.height,
    left: 54 * dpr,
    right: 16 * dpr,
    top: 16 * dpr,
    bottom: 42 * dpr,
  };
  project();
  draw();
}

function project() {
  if (!data || !layout) return;
  const { xlim, ylim } = data;
  const { w, h, left, right, top, bottom } = layout;
  const iw = w - left - right;
  const ih = h - top - bottom;
  sx = data.x.map((v) => left + ((v - xlim[0]) / (xlim[1] - xlim[0])) * iw);
  sy = data.y.map((v) => top + (1 - (v - ylim[0]) / (ylim[1] - ylim[0])) * ih);
}

function xOf(v) {
  const { xlim } = data;
  const { w, left, right } = layout;
  return left + ((v - xlim[0]) / (xlim[1] - xlim[0])) * (w - left - right);
}
function yOf(v) {
  const { ylim } = data;
  const { h, top, bottom } = layout;
  return top + (1 - (v - ylim[0]) / (ylim[1] - ylim[0])) * (h - top - bottom);
}

function drawPoints(indices, color, radius, alpha) {
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.beginPath();
  for (const i of indices) {
    ctx.moveTo(sx[i] + radius, sy[i]);
    ctx.arc(sx[i], sy[i], radius, 0, Math.PI * 2);
  }
  ctx.fill();
  ctx.globalAlpha = 1;
}

function draw() {
  if (!data || !layout) return;
  const { dpr, w, h, left, bottom } = layout;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fffcf8";
  ctx.fillRect(0, 0, w, h);

  const groups = { 0: [], 1: [], 2: [], 3: [], 4: [] };
  for (let i = 0; i < data.k.length; i++) groups[data.k[i]].push(i);

  const r = 2.15 * dpr;
  const dim = 0.16;
  drawPoints(groups[0], "#c9c5bd", 1.7 * dpr, selected ? 0.28 : 0.45);
  for (const id of [1, 2, 3, 4]) {
    const on = !selected || selected === id || (selected <= 2 && id <= 2 && selected === id);
    const active = selected === id;
    drawPoints(groups[id], CELLS[id].color, active ? r + 0.4 * dpr : r, active ? 0.72 : (selected ? dim : 0.5));
  }

  ctx.save();
  ctx.strokeStyle = "#6d685f";
  ctx.lineWidth = 1 * dpr;
  ctx.setLineDash([4 * dpr, 4 * dpr]);
  for (const v of data.px) {
    ctx.beginPath();
    ctx.moveTo(xOf(v), layout.top);
    ctx.lineTo(xOf(v), h - bottom);
    ctx.stroke();
  }
  for (const v of data.py) {
    ctx.beginPath();
    ctx.moveTo(left, yOf(v));
    ctx.lineTo(w - layout.right, yOf(v));
    ctx.stroke();
  }
  ctx.restore();

  ctx.fillStyle = "#1c1b19";
  ctx.font = `${13 * dpr}px "Source Sans 3", sans-serif`;
  ctx.textAlign = "left";
  ctx.fillText("r = 0.378", left + 8 * dpr, layout.top + 18 * dpr);

  ctx.fillStyle = "#5e5a54";
  ctx.font = `${12 * dpr}px "Source Sans 3", sans-serif`;
  ctx.textAlign = "center";
  ctx.fillText("Shared component (PC1)", left + (w - left - layout.right) / 2, h - 12 * dpr);
  ctx.save();
  ctx.translate(16 * dpr, layout.top + (h - layout.top - bottom) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Human vividness", 0, 0);
  ctx.restore();

  if (hover >= 0) {
    ctx.beginPath();
    ctx.arc(sx[hover], sy[hover], 6 * dpr, 0, Math.PI * 2);
    ctx.strokeStyle = "#1c1b19";
    ctx.lineWidth = 1.4 * dpr;
    ctx.stroke();
  }
}

function nearest(mx, my) {
  let best = -1;
  let bestD = (16 * layout.dpr) ** 2;
  for (let i = 0; i < sx.length; i++) {
    const d = (sx[i] - mx) ** 2 + (sy[i] - my) ** 2;
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

canvas.addEventListener("mousemove", (event) => {
  const rect = canvas.getBoundingClientRect();
  const mx = (event.clientX - rect.left) * (canvas.width / rect.width);
  const my = (event.clientY - rect.top) * (canvas.height / rect.height);
  const i = nearest(mx, my);
  if (i !== hover) {
    hover = i;
    if (i >= 0) {
      const cell = Number(data.k[i]);
      const info = CELLS[cell];
      if (cell && cell !== selected) showCell(cell);
      const src = `thumbs/${i}.jpg`;
      if (tipImg.getAttribute("src") !== src) tipImg.src = src;
      tipName.textContent = info.name;
      tipName.style.color = info.color;
      tipMeta.textContent = `PC1 ${data.x[i].toFixed(2)} · vividness ${data.y[i].toFixed(2)}`;
      tip.style.display = "block";
    } else {
      tip.style.display = "none";
    }
    draw();
  }
  if (i >= 0) placeTip(event.clientX, event.clientY);
});

function placeTip(x, y) {
  const pad = 16;
  const tw = tip.offsetWidth || 196;
  const th = tip.offsetHeight || 230;
  let left = x + pad;
  let top = y + pad;
  if (left + tw > window.innerWidth - 8) left = x - tw - pad;
  if (top + th > window.innerHeight - 8) top = Math.max(8, y - th - pad);
  tip.style.left = `${left}px`;
  tip.style.top = `${top}px`;
}

canvas.addEventListener("mouseleave", () => {
  hover = -1;
  tip.style.display = "none";
  draw();
});

chips.forEach((button) => {
  button.addEventListener("click", () => showCell(Number(button.dataset.cell)));
});

window.addEventListener("resize", resize);

fetch("data/scatter.json")
  .then((r) => r.json())
  .then((json) => {
    data = json;
    showCell(3);
    resize();
  });
