const TEAM_CODES = {
  "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "Brentford": "BRE",
  "Brighton": "BHA", "Burnley": "BUR", "Chelsea": "CHE", "Crystal Palace": "CRY",
  "Everton": "EVE", "Fulham": "FUL", "Ipswich": "IPS", "Leeds": "LEE",
  "Leicester": "LEI", "Liverpool": "LIV", "Luton": "LUT", "Man City": "MCI",
  "Man United": "MUN", "Newcastle": "NEW", "Norwich": "NOR", "Nott'm Forest": "NFO",
  "Sheffield United": "SHU", "Southampton": "SOU", "Tottenham": "TOT",
  "Watford": "WAT", "West Brom": "WBA", "West Ham": "WHU", "Wolves": "WOL"
};
const TEAM_COLORS = ["#e11d48", "#2563eb", "#f59e0b", "#10b981", "#8b5cf6",
                     "#06b6d4", "#f97316", "#84cc16", "#ec4899", "#14b8a6"];

function teamColor(name) {
  let h = 0;
  for (const c of name) h = (h * 31 + c.charCodeAt(0)) % 1000;
  return TEAM_COLORS[h % TEAM_COLORS.length];
}

function badge(name, size = 40) {
  const code = TEAM_CODES[name] || name.replace(/[^A-Za-z]/g, "").slice(0, 3).toUpperCase();
  const fs = Math.round(size * 0.3);
  return `<span class="crest" style="width:${size}px;height:${size}px">` +
    `<span class="badge" style="--c:${teamColor(name)};width:100%;height:100%;font-size:${fs}px">${code}</span>` +
    `<img src="/static/img/teams/${code.toLowerCase()}.png" alt="${esc(name)}" loading="lazy" ` +
    `onload="this.previousElementSibling.style.display='none'" onerror="this.remove()">` +
    `</span>`;
}

async function api(url, options) {
  const res = await fetch(url, options);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Lỗi máy chủ");
  return data;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
}