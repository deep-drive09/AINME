"""
ANIMATRIX — Anime Recommendation System (Data Mining Project)
Single File Flask App
─────────────────────────────────────────────────
Install:  pip install flask pandas numpy scikit-learn
Run:      python anime_recommender.py
Open:     http://127.0.0.1:5000
"""

from flask import Flask, render_template_string, request
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import json

app = Flask(__name__)

# ══════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════
ANIME_DATA = [
    {"id":1,  "title":"Naruto",                          "genre":["Action","Adventure","Martial Arts"],  "rating":7.9,"episodes":220, "popularity":1},
    {"id":2,  "title":"Death Note",                      "genre":["Mystery","Psychological","Thriller"], "rating":8.6,"episodes":37,  "popularity":2},
    {"id":3,  "title":"Attack on Titan",                 "genre":["Action","Drama","Fantasy"],           "rating":9.0,"episodes":87,  "popularity":3},
    {"id":4,  "title":"Fullmetal Alchemist Brotherhood", "genre":["Action","Adventure","Fantasy"],       "rating":9.1,"episodes":64,  "popularity":4},
    {"id":5,  "title":"One Piece",                       "genre":["Action","Adventure","Comedy"],        "rating":8.7,"episodes":1000,"popularity":5},
    {"id":6,  "title":"Demon Slayer",                    "genre":["Action","Fantasy","Historical"],      "rating":8.7,"episodes":44,  "popularity":6},
    {"id":7,  "title":"Dragon Ball Z",                   "genre":["Action","Adventure","Martial Arts"],  "rating":8.3,"episodes":291, "popularity":7},
    {"id":8,  "title":"Sword Art Online",                "genre":["Action","Adventure","Romance"],       "rating":7.2,"episodes":25,  "popularity":8},
    {"id":9,  "title":"My Hero Academia",                "genre":["Action","Comedy","School"],           "rating":7.9,"episodes":113, "popularity":9},
    {"id":10, "title":"Tokyo Ghoul",                     "genre":["Action","Drama","Horror"],            "rating":7.8,"episodes":12,  "popularity":10},
    {"id":11, "title":"Hunter x Hunter",                 "genre":["Action","Adventure","Fantasy"],       "rating":9.0,"episodes":148, "popularity":11},
    {"id":12, "title":"Steins Gate",                     "genre":["Drama","Psychological","Sci-Fi"],     "rating":9.1,"episodes":24,  "popularity":12},
    {"id":13, "title":"Code Geass",                      "genre":["Action","Drama","Mecha"],             "rating":8.7,"episodes":25,  "popularity":13},
    {"id":14, "title":"Neon Genesis Evangelion",         "genre":["Drama","Mecha","Psychological"],      "rating":8.5,"episodes":26,  "popularity":14},
    {"id":15, "title":"Cowboy Bebop",                    "genre":["Action","Adventure","Sci-Fi"],        "rating":8.9,"episodes":26,  "popularity":15},
    {"id":16, "title":"Spirited Away",                   "genre":["Adventure","Fantasy","Supernatural"], "rating":8.8,"episodes":1,   "popularity":16},
    {"id":17, "title":"Your Lie in April",               "genre":["Drama","Music","Romance"],            "rating":8.7,"episodes":22,  "popularity":17},
    {"id":18, "title":"Violet Evergarden",               "genre":["Drama","Fantasy","Romance"],          "rating":8.7,"episodes":13,  "popularity":18},
    {"id":19, "title":"Re Zero",                         "genre":["Drama","Fantasy","Psychological"],    "rating":8.3,"episodes":50,  "popularity":19},
    {"id":20, "title":"Bleach",                          "genre":["Action","Adventure","Supernatural"],  "rating":7.9,"episodes":366, "popularity":20},
    {"id":21, "title":"Fairy Tail",                      "genre":["Action","Adventure","Comedy"],        "rating":7.6,"episodes":328, "popularity":21},
    {"id":22, "title":"Black Clover",                    "genre":["Action","Adventure","Fantasy"],       "rating":7.9,"episodes":170, "popularity":22},
    {"id":23, "title":"Jujutsu Kaisen",                  "genre":["Action","Fantasy","School"],          "rating":8.7,"episodes":24,  "popularity":23},
    {"id":24, "title":"Vinland Saga",                    "genre":["Action","Adventure","Historical"],    "rating":8.7,"episodes":24,  "popularity":24},
    {"id":25, "title":"Made in Abyss",                   "genre":["Adventure","Drama","Fantasy"],        "rating":8.7,"episodes":13,  "popularity":25},
    {"id":26, "title":"Haikyuu",                         "genre":["Comedy","School","Sports"],           "rating":8.7,"episodes":85,  "popularity":26},
    {"id":27, "title":"Kurokos Basketball",              "genre":["Comedy","School","Sports"],           "rating":8.1,"episodes":75,  "popularity":27},
    {"id":28, "title":"Slam Dunk",                       "genre":["Comedy","Drama","Sports"],            "rating":8.5,"episodes":101, "popularity":28},
    {"id":29, "title":"Toradora",                        "genre":["Comedy","Romance","School"],          "rating":8.1,"episodes":25,  "popularity":29},
    {"id":30, "title":"Clannad After Story",             "genre":["Drama","Fantasy","Romance"],          "rating":8.9,"episodes":24,  "popularity":30},
]

WATCH_HISTORY = [
    ["Naruto","Dragon Ball Z","Bleach","One Piece"],
    ["Death Note","Code Geass","Steins Gate"],
    ["Attack on Titan","Demon Slayer","Jujutsu Kaisen","Vinland Saga"],
    ["Fullmetal Alchemist Brotherhood","Hunter x Hunter","Attack on Titan"],
    ["One Piece","Naruto","Fairy Tail","Black Clover"],
    ["Sword Art Online","Re Zero","Steins Gate"],
    ["My Hero Academia","Naruto","Demon Slayer","Black Clover"],
    ["Tokyo Ghoul","Attack on Titan","Jujutsu Kaisen"],
    ["Hunter x Hunter","Fullmetal Alchemist Brotherhood","Cowboy Bebop"],
    ["Steins Gate","Death Note","Neon Genesis Evangelion"],
    ["Code Geass","Death Note","Neon Genesis Evangelion"],
    ["Your Lie in April","Violet Evergarden","Clannad After Story","Toradora"],
    ["Haikyuu","Kurokos Basketball","Slam Dunk"],
    ["Re Zero","Made in Abyss","Violet Evergarden"],
    ["Spirited Away","Made in Abyss","Violet Evergarden"],
    ["Cowboy Bebop","Steins Gate","Neon Genesis Evangelion"],
    ["Dragon Ball Z","Naruto","One Piece","Bleach"],
    ["Demon Slayer","Jujutsu Kaisen","My Hero Academia"],
    ["Clannad After Story","Your Lie in April","Toradora","Violet Evergarden"],
    ["Vinland Saga","Attack on Titan","Made in Abyss"],
    ["Jujutsu Kaisen","Demon Slayer","Attack on Titan","My Hero Academia"],
    ["Toradora","Your Lie in April","Clannad After Story"],
    ["Bleach","Naruto","Dragon Ball Z","Fairy Tail"],
    ["Fullmetal Alchemist Brotherhood","Attack on Titan","Hunter x Hunter","Code Geass"],
    ["Sword Art Online","Re Zero","Black Clover"],
]

POSTERS = {
    "Naruto":                          "https://cdn.myanimelist.net/images/anime/13/17405.jpg",
    "Death Note":                      "https://cdn.myanimelist.net/images/anime/9/9453.jpg",
    "Attack on Titan":                 "https://cdn.myanimelist.net/images/anime/10/47347.jpg",
    "Fullmetal Alchemist Brotherhood": "https://cdn.myanimelist.net/images/anime/1223/96541.jpg",
    "One Piece":                       "https://cdn.myanimelist.net/images/anime/6/73245.jpg",
    "Demon Slayer":                    "https://cdn.myanimelist.net/images/anime/1286/99889.jpg",
    "Dragon Ball Z":                   "https://cdn.myanimelist.net/images/anime/5/19571.jpg",
    "Sword Art Online":                "https://cdn.myanimelist.net/images/anime/11/39717.jpg",
    "My Hero Academia":                "https://cdn.myanimelist.net/images/anime/10/78745.jpg",
    "Tokyo Ghoul":                     "https://cdn.myanimelist.net/images/anime/5/64449.jpg",
    "Hunter x Hunter":                 "https://cdn.myanimelist.net/images/anime/11/33657.jpg",
    "Steins Gate":                     "https://cdn.myanimelist.net/images/anime/5/73199.jpg",
    "Code Geass":                      "https://cdn.myanimelist.net/images/anime/5/50331.jpg",
    "Neon Genesis Evangelion":         "https://cdn.myanimelist.net/images/anime/7/20310.jpg",
    "Cowboy Bebop":                    "https://cdn.myanimelist.net/images/anime/4/19644.jpg",
    "Spirited Away":                   "https://cdn.myanimelist.net/images/anime/6/79597.jpg",
    "Your Lie in April":               "https://cdn.myanimelist.net/images/anime/3/67177.jpg",
    "Violet Evergarden":               "https://cdn.myanimelist.net/images/anime/1795/95088.jpg",
    "Re Zero":                         "https://cdn.myanimelist.net/images/anime/11/79410.jpg",
    "Bleach":                          "https://cdn.myanimelist.net/images/anime/3/40451.jpg",
    "Fairy Tail":                      "https://cdn.myanimelist.net/images/anime/5/18179.jpg",
    "Black Clover":                    "https://cdn.myanimelist.net/images/anime/6/90033.jpg",
    "Jujutsu Kaisen":                  "https://cdn.myanimelist.net/images/anime/1171/109222.jpg",
    "Vinland Saga":                    "https://cdn.myanimelist.net/images/anime/1500/103005.jpg",
    "Made in Abyss":                   "https://cdn.myanimelist.net/images/anime/6/86733.jpg",
    "Haikyuu":                         "https://cdn.myanimelist.net/images/anime/7/76014.jpg",
    "Kurokos Basketball":              "https://cdn.myanimelist.net/images/anime/3/43049.jpg",
    "Slam Dunk":                       "https://cdn.myanimelist.net/images/anime/1804/90389.jpg",
    "Toradora":                        "https://cdn.myanimelist.net/images/anime/13/22128.jpg",
    "Clannad After Story":             "https://cdn.myanimelist.net/images/anime/1302/75887.jpg",
}

df = pd.DataFrame(ANIME_DATA)
ALL_GENRES = sorted(set(g for row in df['genre'] for g in row))
ALL_TITLES = df['title'].tolist()

def poster(title):
    return POSTERS.get(title, "https://placehold.co/200x290/111827/e94560?text=No+Poster")

# ══════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════
CSS = """
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#080c14;--bg2:#0e1420;--bg3:#111827;--card:#0f1929;
  --red:#e94560;--gold:#f59e0b;--cyan:#22d3ee;
  --text:#eef0f8;--muted:#6b7280;--border:#1e2d45;
  --font:'Outfit',sans-serif;--disp:'Bebas Neue',sans-serif;
}
body{background:var(--bg);color:var(--text);font-family:var(--font);min-height:100vh}
a{text-decoration:none;color:inherit}
nav{background:rgba(8,12,20,.97);border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  padding:0 40px;height:62px;position:sticky;top:0;z-index:999}
.brand{display:flex;align-items:center;gap:10px}
.bn{font-family:var(--disp);font-size:28px;color:var(--red);letter-spacing:3px}
.bs{font-size:10px;color:var(--muted);letter-spacing:2px;font-weight:700}
.nl{display:flex;gap:4px}
.nl a{color:var(--muted);padding:8px 16px;border-radius:8px;font-weight:700;font-size:13px;
  transition:all .2s;border:1px solid transparent}
.nl a:hover{color:var(--text);background:var(--bg3);border-color:var(--border)}
.nl a.on{color:#fff;background:var(--red);border-color:var(--red)}
.wrap{max-width:1200px;margin:0 auto;padding:36px 24px}
.card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:28px;margin-bottom:24px}
.st{font-family:var(--disp);font-size:30px;letter-spacing:2px;color:#fff;margin-bottom:20px}
.ph{text-align:center;padding:48px 0 28px}
.ph h1{font-family:var(--disp);font-size:58px;letter-spacing:4px;color:#fff;line-height:1}
.ph p{color:var(--muted);font-size:15px;margin-top:10px;font-weight:600}
.red{color:var(--red)}.gold{color:var(--gold)}.cyan{color:var(--cyan)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(172px,1fr));gap:20px}
.ac{background:var(--bg3);border-radius:14px;overflow:hidden;border:1px solid var(--border);transition:transform .25s,box-shadow .25s}
.ac:hover{transform:translateY(-6px);box-shadow:0 16px 40px rgba(0,0,0,.6)}
.pw{position:relative;overflow:hidden}
.ap{width:100%;height:252px;object-fit:cover;display:block;transition:transform .3s}
.ac:hover .ap{transform:scale(1.06)}
.rb{position:absolute;top:8px;right:8px;background:rgba(0,0,0,.88);border:1px solid rgba(255,255,255,.1);padding:3px 9px;border-radius:50px;font-size:12px;font-weight:700}
.ai{padding:12px}
.at{font-size:13px;font-weight:800;margin-bottom:4px;line-height:1.3;color:#fff}
.ag{font-size:11px;color:var(--muted);margin-bottom:4px;line-height:1.4}
.ae{font-size:11px;color:var(--muted)}
.btn{background:var(--red);color:#fff;border:none;padding:13px 38px;border-radius:50px;
  font-size:15px;font-weight:800;cursor:pointer;font-family:var(--font);letter-spacing:.5px;
  transition:transform .2s,box-shadow .2s;display:inline-block}
.btn:hover{transform:translateY(-2px);box-shadow:0 8px 28px rgba(233,69,96,.4)}
.sm{display:inline-block;background:var(--red);color:#fff;border:none;padding:5px 13px;
  border-radius:50px;font-size:11px;font-weight:700;cursor:pointer;font-family:var(--font);
  transition:opacity .2s;margin-top:6px}
.sm:hover{opacity:.8}
.sm.o{background:transparent;border:2px solid var(--red);color:var(--red)}
.sm.o:hover{background:var(--red);color:#fff;opacity:1}
select{background:var(--bg3);color:var(--text);border:1px solid var(--border);border-radius:8px;
  padding:9px 14px;font-family:var(--font);font-size:14px;font-weight:600;cursor:pointer;outline:none;transition:border-color .2s}
select:focus{border-color:var(--red)}
.chips{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:22px}
.chip{display:inline-flex;align-items:center;gap:6px;padding:8px 18px;border-radius:50px;
  border:2px solid var(--border);background:var(--bg3);color:var(--muted);cursor:pointer;
  font-weight:700;font-size:13px;transition:all .2s;user-select:none}
.chip input{display:none}
.chip:hover{border-color:var(--red);color:var(--red)}
.chip.on{border-color:var(--red);background:var(--red);color:#fff}
.banner{display:flex;align-items:center;gap:24px;flex-wrap:wrap}
.bp{width:115px;height:163px;object-fit:cover;border-radius:10px;flex-shrink:0;border:2px solid var(--border)}
.bi2 h2{font-family:var(--disp);font-size:34px;letter-spacing:1px;color:#fff}
.bi2 p{color:var(--muted);margin-top:5px;font-size:14px;font-weight:600}
.bdg{display:inline-block;background:var(--red);color:#fff;padding:4px 14px;border-radius:50px;font-size:11px;font-weight:700;margin-top:10px}
.mr{display:flex;gap:6px;margin-top:7px;flex-wrap:wrap}
.mp{background:var(--bg2);border:1px solid var(--border);padding:3px 9px;border-radius:50px;font-size:11px;font-weight:700;color:var(--muted)}
.mp.g{border-color:var(--gold);color:var(--gold)}
.mp.c{border-color:var(--cyan);color:var(--cyan)}
.stats{display:flex;justify-content:space-around;flex-wrap:wrap;gap:16px;
  background:var(--card);border:1px solid var(--border);border-radius:16px;padding:28px;margin-top:28px}
.si{text-align:center}
.sn{display:block;font-family:var(--disp);font-size:52px;color:var(--red);line-height:1}
.sl{font-size:12px;color:var(--muted);font-weight:700;letter-spacing:1px;text-transform:uppercase}
.abox h3{font-family:var(--disp);font-size:22px;letter-spacing:1px;margin-bottom:16px;color:#fff}
.steps{display:flex;gap:16px;flex-wrap:wrap}
.step{display:flex;align-items:flex-start;gap:10px;flex:1;min-width:180px}
.sno{background:var(--red);color:#fff;width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:13px;flex-shrink:0}
.step p{font-size:13px;color:var(--muted);line-height:1.6}
.cl{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:20px}
.cpil{display:flex;align-items:center;gap:8px;padding:7px 16px;border-radius:50px;border:1px solid var(--border);cursor:pointer;font-size:13px;font-weight:700;transition:all .2s}
.cpil:hover{border-color:var(--red)}
.cdot{width:12px;height:12px;border-radius:50%;flex-shrink:0}
.ch{display:flex;align-items:center;justify-content:space-between;padding-left:14px;margin-bottom:16px}
.ch h2{font-family:var(--disp);font-size:24px;letter-spacing:1px;color:#fff}
.cc{background:var(--bg2);padding:4px 13px;border-radius:50px;font-size:13px;font-weight:700;color:var(--muted)}
.db{height:3px;background:var(--border)}
.df{height:100%;background:linear-gradient(90deg,var(--red),var(--gold));transition:width .5s}
.err{color:var(--red);font-weight:700;margin-bottom:14px;font-size:14px}
.scatter-wrap{position:relative;width:100%;background:var(--bg2);border-radius:12px;padding:8px;border:1px solid var(--border)}
footer{text-align:center;padding:24px;color:var(--muted);font-size:12px;border-top:1px solid var(--border);margin-top:16px;font-weight:600;letter-spacing:.5px}
"""

def page(title, active, body):
    links = [("/","dash","🏠 Dashboard"),("/association","assoc","🔗 Association"),
             ("/clustering","clust","🌐 Clustering"),("/classification","klass","🤖 Classification")]
    nav = "".join(f'<a href="{h}" class="{"on" if k==active else ""}">{l}</a>' for h,k,l in links)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Animatrix — {title}</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<nav>
  <div class="brand">
    <span style="font-size:26px">⛩</span>
    <div><div class="bn">ANIMATRIX</div><div class="bs">DATA MINING SYSTEM</div></div>
  </div>
  <div class="nl">{nav}</div>
</nav>
{body}
<footer>Animatrix &mdash; Anime Recommendation System &nbsp;|&nbsp; Data Mining Project &copy; 2025 &nbsp;|&nbsp; Flask + Scikit-learn</footer>
</body></html>"""


# ══════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════
DASH_T = """
<div class="wrap">
  <div style="text-align:center;padding:56px 0 36px">
    <div style="font-family:'Bebas Neue',sans-serif;font-size:78px;line-height:.95;letter-spacing:4px;color:#fff">
      DISCOVER YOUR<br><span class="red">NEXT ANIME</span>
    </div>
    <p style="color:#6b7280;font-size:17px;margin-top:14px;font-weight:600">
      Association Rules &bull; K-Means Clustering &bull; KNN Classification
    </p>
  </div>
  <div class="card">
    <div class="st">🎭 Select Your Genres</div>
    <form method="POST" action="/recommend">
      <div class="chips">
        {% for g in genres %}
        <label class="chip {% if g in sel %}on{% endif %}">
          <input type="checkbox" name="genres" value="{{ g }}" {% if g in sel %}checked{% endif %}>{{ g }}
        </label>
        {% endfor %}
      </div>
      {% if error %}<p class="err">⚠ {{ error }}</p>{% endif %}
      <button type="submit" class="btn">🔍 Get Recommendations</button>
    </form>
  </div>
  {% if recs %}
  <div class="st">🌟 Top Picks for You</div>
  <div class="grid">
    {% for a in recs %}
    <div class="ac">
      <div class="pw">
        <img src="{{ a.poster }}" class="ap" alt="{{ a.title }}"
             onerror="this.src='https://placehold.co/200x290/111827/e94560?text=No+Poster'">
        <div class="rb">⭐ {{ a.rating }}</div>
      </div>
      <div class="ai">
        <div class="at">{{ a.title }}</div>
        <div class="ag">{{ a.genre }}</div>
        <div class="ae">📺 {{ a.episodes }} eps</div>
        <div style="display:flex;gap:6px;margin-top:8px;flex-wrap:wrap">
          <a href="/association?anime={{ a.title }}" class="sm">Associations</a>
          <a href="/classification?anime={{ a.title }}" class="sm o">Classify</a>
        </div>
      </div>
    </div>
    {% endfor %}
  </div>
  {% endif %}
  <div class="stats">
    <div class="si"><span class="sn">30</span><span class="sl">Anime Titles</span></div>
    <div class="si"><span class="sn">15</span><span class="sl">Genres</span></div>
    <div class="si"><span class="sn">4</span><span class="sl">ML Algorithms</span></div>
    <div class="si"><span class="sn">25</span><span class="sl">Watch Histories</span></div>
  </div>
</div>
"""

@app.route('/')
def dashboard():
    body = render_template_string(DASH_T, genres=ALL_GENRES, sel=[], recs=[], error=None)
    return page("Dashboard", "dash", body)

@app.route('/recommend', methods=['POST'])
def recommend():
    sel = request.form.getlist('genres')
    if not sel:
        body = render_template_string(DASH_T, genres=ALL_GENRES, sel=[], recs=[], error="Please select at least one genre!")
        return page("Dashboard", "dash", body)
    df['_s'] = df['genre'].apply(lambda g: len(set(g) & set(sel)))
    top = df[df['_s'] > 0].sort_values(['_s','rating'], ascending=False).head(5)
    recs = [{"title":r['title'],"genre":", ".join(r['genre']),"rating":r['rating'],
             "episodes":r['episodes'],"poster":poster(r['title'])} for _,r in top.iterrows()]
    body = render_template_string(DASH_T, genres=ALL_GENRES, sel=sel, recs=recs, error=None)
    return page("Dashboard", "dash", body)


# ══════════════════════════════════════════════════════════
# ASSOCIATION
# ══════════════════════════════════════════════════════════
ASSOC_T = """
<div class="wrap">
  <div class="ph">
    <h1>🔗 ASSOCIATION<br><span class="red">MINING</span></h1>
    <p>Co-occurrence Analysis &bull; Support &bull; Confidence &bull; Lift</p>
  </div>
  <div class="card" style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
    <span style="font-weight:700;color:#6b7280">🎌 Select Anime:</span>
    <select onchange="window.location.href='/association?anime='+encodeURIComponent(this.value)">
      {% for t in titles %}
      <option value="{{ t }}" {% if t==sel %}selected{% endif %}>{{ t }}</option>
      {% endfor %}
    </select>
  </div>
  <div class="card banner">
    <img src="{{ sd.poster }}" class="bp" alt="{{ sd.title }}"
         onerror="this.src='https://placehold.co/115x163/111827/e94560?text=N/A'">
    <div class="bi2">
      <h2>{{ sd.title }}</h2>
      <p>🎭 {{ sd.genre }}</p>
      <p>⭐ Rating: {{ sd.rating }}</p>
      <p style="margin-top:8px;font-size:13px;color:#eef0f8">
        Mined from <strong style="color:#f59e0b">25 watch-history transactions</strong>
      </p>
      <div class="bdg">Co-occurrence: Support · Confidence · Lift</div>
    </div>
  </div>
  {% if assoc %}
  <p style="color:#6b7280;font-weight:700;font-size:15px;margin-bottom:18px">
    👥 People who watched <span style="color:#e94560">{{ sel }}</span> also watched:
  </p>
  <div class="grid">
    {% for a in assoc %}
    <div class="ac" style="max-width:140px">
      <div class="pw">
        <img src="{{ a.poster }}" class="ap" alt="{{ a.title }}" style="height:190px"
             onerror="this.src='https://placehold.co/200x290/111827/e94560?text=No+Poster'">
        <div class="rb" style="font-size:10px">⭐ {{ a.rating }}</div>
      </div>
      <div class="ai" style="padding:8px">
        <div class="at" style="font-size:11px">{{ a.title }}</div>
        <div class="mr">
          <span class="mp" style="font-size:10px">Conf: {{ a.confidence }}</span>
          <span class="mp g" style="font-size:10px">Lift: {{ a.lift }}</span>
        </div>
        <a href="/association?anime={{ a.title }}" class="sm" style="margin-top:6px;font-size:10px">Explore →</a>
      </div>
    </div>
    {% endfor %}
  </div>
  {% else %}
  <div class="card" style="text-align:center;padding:48px;color:#6b7280">No associations found.</div>
  {% endif %}
</div>
"""

@app.route('/association')
def association():
    sel = request.args.get('anime', 'Attack on Titan')
    total = len(WATCH_HISTORY)
    item_cnt = defaultdict(int)
    co_cnt   = defaultdict(int)
    for session in WATCH_HISTORY:
        uniq = list(set(session))
        for t in uniq:
            item_cnt[t] += 1
        for i in range(len(uniq)):
            for j in range(len(uniq)):
                if i != j:
                    co_cnt[(uniq[i], uniq[j])] += 1

    seen, assoc = set(), []
    if sel in item_cnt:
        pairs = [(b, co_cnt[(sel,b)]) for b in item_cnt if b != sel and co_cnt[(sel,b)] > 0]
        pairs.sort(key=lambda x: -x[1])
        for item, freq in pairs:
            sa  = item_cnt[sel]/total
            sb  = item_cnt[item]/total
            sab = freq/total
            conf = round(sab/sa, 2) if sa else 0
            lift = round(sab/(sa*sb), 2) if sa*sb else 0
            row  = df[df['title']==item]
            if not row.empty and item not in seen:
                seen.add(item)
                r = row.iloc[0]
                assoc.append({"title":item,"genre":", ".join(r['genre']),
                              "rating":r['rating'],"poster":poster(item),
                              "confidence":conf,"lift":lift})

    if len(assoc) < 4:
        si = df[df['title']==sel]
        if not si.empty:
            sg = set(si.iloc[0]['genre'])
            scored = []
            for _, r in df[df['title']!=sel].iterrows():
                if r['title'] not in seen:
                    ov = len(set(r['genre']) & sg)
                    if ov:
                        scored.append({"title":r['title'],"genre":", ".join(r['genre']),
                                       "rating":r['rating'],"poster":poster(r['title']),
                                       "confidence":round(ov/len(sg),2),"lift":round(float(ov),2)})
            scored.sort(key=lambda x: -x['lift'])
            assoc += scored[:8-len(assoc)]

    sr = df[df['title']==sel].iloc[0] if not df[df['title']==sel].empty else df.iloc[0]
    sd = {"title":sel,"poster":poster(sel),"genre":", ".join(sr['genre']),"rating":sr['rating']}
    body = render_template_string(ASSOC_T, titles=ALL_TITLES, sel=sel, sd=sd, assoc=assoc[:8])
    return page("Association", "assoc", body)


# ══════════════════════════════════════════════════════════
# CLUSTERING  — FIX: scatter_json passed via Python string
#               interpolation (not Jinja), so no escaping
# ══════════════════════════════════════════════════════════
CLUST_T_TOP = """
<div class="wrap">
  <div class="ph">
    <h1>🌐 GENRE<br><span class="red">CLUSTERING</span></h1>
    <p>K-Means (k=5) on genre vectors &bull; PCA 2D visualisation</p>
  </div>
  <div class="card">
    <div class="st" style="margin-bottom:12px">📈 PCA Scatter Plot</div>
    <div class="cl">
      {LEGEND}
    </div>
    <div class="scatter-wrap">
      <canvas id="sc" style="width:100%;height:420px"></canvas>
    </div>
  </div>
  {CLUSTER_CARDS}
  <div class="card abox">
    <h3>📊 How K-Means Clustering Works</h3>
    <div class="steps">
      <div class="step"><div class="sno">1</div><p>One-hot encode anime genres using <strong>MultiLabelBinarizer</strong></p></div>
      <div class="step"><div class="sno">2</div><p>Apply <strong>K-Means (k=5)</strong> on the genre feature matrix</p></div>
      <div class="step"><div class="sno">3</div><p>Reduce to 2D using <strong>PCA</strong> for scatter visualisation</p></div>
      <div class="step"><div class="sno">4</div><p>Anime in same cluster share <strong>genre similarity patterns</strong></p></div>
    </div>
  </div>
</div>
"""

CLUST_SCRIPT = """
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
(function(){
  var raw = SCATTER_DATA_PLACEHOLDER;
  var dsMap = {};
  raw.forEach(function(p){
    if(!dsMap[p.cluster]){
      dsMap[p.cluster] = {
        label: p.name,
        data: [],
        backgroundColor: p.color,
        pointRadius: 9,
        pointHoverRadius: 13,
        pointBorderWidth: 2,
        pointBorderColor: 'rgba(255,255,255,0.2)'
      };
    }
    dsMap[p.cluster].data.push({x: p.x, y: p.y, title: p.title});
  });
  var datasets = Object.keys(dsMap).sort().map(function(k){ return dsMap[k]; });
  var ctx = document.getElementById('sc').getContext('2d');
  new Chart(ctx, {
    type: 'scatter',
    data: { datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 800 },
      plugins: {
        legend: {
          display: true,
          labels: { color: '#9ca3af', font: { family: 'Outfit', size: 12 }, padding: 16 }
        },
        tooltip: {
          callbacks: {
            label: function(c){ return '  ' + c.raw.title; }
          }
        }
      },
      scales: {
        x: {
          ticks: { color: '#4b5563' },
          grid: { color: 'rgba(30,45,69,0.8)' },
          title: { display: true, text: 'PCA Component 1', color: '#6b7280' }
        },
        y: {
          ticks: { color: '#4b5563' },
          grid: { color: 'rgba(30,45,69,0.8)' },
          title: { display: true, text: 'PCA Component 2', color: '#6b7280' }
        }
      }
    }
  });
  function jump(id){
    document.querySelectorAll('[id^=cl]').forEach(function(e){ e.style.opacity='0.3'; });
    var el = document.getElementById('cl'+id);
    if(el){ el.style.opacity='1'; el.scrollIntoView({behavior:'smooth',block:'start'}); }
    setTimeout(function(){ document.querySelectorAll('[id^=cl]').forEach(function(e){ e.style.opacity='1'; }); }, 2500);
  }
  window.jump = jump;
})();
</script>
"""

@app.route('/clustering')
def clustering():
    mlb  = MultiLabelBinarizer()
    gmat = mlb.fit_transform(df['genre'])
    km   = KMeans(n_clusters=5, random_state=42, n_init=10)
    lbls = km.fit_predict(gmat)
    pca  = PCA(n_components=2, random_state=42)
    xy   = pca.fit_transform(gmat)

    COLORS = ['#e94560','#22d3ee','#f59e0b','#a78bfa','#34d399']
    NAMES  = {0:"Action / Adventure",1:"Sci-Fi / Psychological",
              2:"Drama / Romance",3:"Sports / Comedy",4:"Fantasy / Historical"}

    tmp = df.copy()
    tmp['_c']  = lbls
    tmp['_px'] = xy[:,0]
    tmp['_py'] = xy[:,1]

    # Build scatter JSON (plain Python, no Jinja)
    scatter = []
    for _, r in tmp.iterrows():
        c = int(r['_c'])
        scatter.append({
            "title": r['title'],
            "x": round(float(r['_px']), 3),
            "y": round(float(r['_py']), 3),
            "cluster": c,
            "color": COLORS[c],
            "name": NAMES[c]
        })

    # Legend HTML
    legend_html = ""
    for c in range(5):
        cnt = int((tmp['_c'] == c).sum())
        legend_html += (
            f'<div class="cpil" onclick="jump({c})">'
            f'<div class="cdot" style="background:{COLORS[c]}"></div>'
            f'<span>{NAMES[c]}</span>'
            f'<span style="color:#6b7280;font-size:11px">({cnt})</span>'
            f'</div>'
        )

    # Cluster card HTML
    cards_html = ""
    for c in range(5):
        sub = tmp[tmp['_c'] == c]
        anime_cards = ""
        for _, r in sub.iterrows():
            anime_cards += f"""
            <div class="ac">
              <div style="height:3px;background:{COLORS[c]}"></div>
              <div class="pw">
                <img src="{poster(r['title'])}" class="ap" alt="{r['title']}"
                     onerror="this.src='https://placehold.co/200x290/111827/e94560?text=No+Poster'">
                <div class="rb">⭐ {r['rating']}</div>
              </div>
              <div class="ai">
                <div class="at">{r['title']}</div>
                <div class="ag">{', '.join(r['genre'])}</div>
              </div>
            </div>"""
        cards_html += f"""
        <div class="card" id="cl{c}" style="border-left:4px solid {COLORS[c]}">
          <div class="ch">
            <h2>{NAMES[c]}</h2>
            <div class="cc">{len(sub)} Anime</div>
          </div>
          <div class="grid">{anime_cards}</div>
        </div>"""

    # Inject scatter JSON directly into the JS — no Jinja involved
    scatter_json_str = json.dumps(scatter)
    script = CLUST_SCRIPT.replace("SCATTER_DATA_PLACEHOLDER", scatter_json_str)

    body_html = (
        CLUST_T_TOP
        .replace("{LEGEND}", legend_html)
        .replace("{CLUSTER_CARDS}", cards_html)
        + script
    )
    return page("Clustering", "clust", body_html)


# ══════════════════════════════════════════════════════════
# CLASSIFICATION
# ══════════════════════════════════════════════════════════
CLASS_T = """
<div class="wrap">
  <div class="ph">
    <h1>🤖 KNN<br><span class="red">CLASSIFICATION</span></h1>
    <p>K-Nearest Neighbors &bull; Euclidean Distance &bull; Genre + Rating + Episodes features</p>
  </div>
  <div class="card" style="display:flex;gap:20px;align-items:center;flex-wrap:wrap">
    <div style="display:flex;align-items:center;gap:10px">
      <span style="color:#6b7280;font-weight:700">🎌 Anime:</span>
      <select id="as" onchange="go()">
        {% for t in titles %}
        <option value="{{ t }}" {% if t==sel %}selected{% endif %}>{{ t }}</option>
        {% endfor %}
      </select>
    </div>
    <div style="display:flex;align-items:center;gap:10px">
      <span style="color:#6b7280;font-weight:700">K Value:</span>
      <select id="ks" onchange="go()">
        {% for kv in [3,5,7,10] %}
        <option value="{{ kv }}" {% if kv==k %}selected{% endif %}>K = {{ kv }}</option>
        {% endfor %}
      </select>
    </div>
  </div>
  <div class="card banner">
    <img src="{{ sd.poster }}" class="bp" alt="{{ sd.title }}"
         onerror="this.src='https://placehold.co/115x163/111827/e94560?text=N/A'">
    <div class="bi2">
      <h2>{{ sd.title }}</h2>
      <p>🎭 {{ sd.genre }}</p>
      <p>⭐ Rating: {{ sd.rating }}</p>
      <p style="margin-top:6px">📂 Predicted Class: <strong class="red">{{ sd.pred }}</strong></p>
      <div class="bdg">KNN · k={{ k }} · Euclidean Distance</div>
    </div>
  </div>
  <div class="st">📍 {{ k }} Nearest Neighbors</div>
  <div class="grid">
    {% for a in neighbors %}
    <div class="ac">
      <div class="db"><div class="df" style="width:{{ a.bar_pct }}%"></div></div>
      <div class="pw">
        <img src="{{ a.poster }}" class="ap" alt="{{ a.title }}"
             onerror="this.src='https://placehold.co/200x290/111827/e94560?text=No+Poster'">
        <div class="rb">⭐ {{ a.rating }}</div>
      </div>
      <div class="ai">
        <div class="at">{{ a.title }}</div>
        <div class="ag">{{ a.genre }}</div>
        <div class="mr">
          <span class="mp c">Dist: {{ a.dist }}</span>
          <span class="mp g">{{ a.pred }}</span>
        </div>
        <div class="ae" style="margin-top:5px">📺 {{ a.episodes }} eps</div>
      </div>
    </div>
    {% endfor %}
  </div>
  <div class="card abox" style="margin-top:8px">
    <h3>📊 How KNN Classification Works</h3>
    <div class="steps">
      <div class="step"><div class="sno">1</div><p>Feature vector = <strong>Genre (one-hot)</strong> + Rating + Episodes + Popularity</p></div>
      <div class="step"><div class="sno">2</div><p>Compute <strong>Euclidean distance</strong> from query to every other anime</p></div>
      <div class="step"><div class="sno">3</div><p>Pick <strong>K nearest neighbors</strong> with smallest distances</p></div>
      <div class="step"><div class="sno">4</div><p>Classify by <strong>majority vote</strong> of neighbors' genre labels</p></div>
    </div>
  </div>
</div>
<script>
function go(){
  var a=document.getElementById('as').value;
  var k=document.getElementById('ks').value;
  window.location.href='/classification?anime='+encodeURIComponent(a)+'&k='+k;
}
</script>
"""

@app.route('/classification')
def classification():
    sel = request.args.get('anime', 'Naruto')
    k   = int(request.args.get('k', 5))

    mlb   = MultiLabelBinarizer()
    gmat  = mlb.fit_transform(df['genre'])
    feats = np.hstack([gmat, df[['rating','episodes','popularity']].values.astype(float)])

    le = LabelEncoder()
    y  = le.fit_transform(df['genre'].apply(lambda x: x[0]))

    knn = KNeighborsClassifier(n_neighbors=k+1, metric='euclidean')
    knn.fit(feats, y)

    idx_s = df[df['title']==sel].index
    idx   = idx_s[0] if len(idx_s) else 0

    dists, idxs = knn.kneighbors(feats[idx].reshape(1,-1), n_neighbors=k+1)
    max_dist = float(dists[0][-1]) if dists[0][-1] > 0 else 1.0

    neighbors = []
    for i, ni in enumerate(idxs[0]):
        if df.iloc[ni]['title'] == sel:
            continue
        r = df.iloc[ni]
        d = float(dists[0][i])
        bar = max(8, int(100 - (d / max_dist) * 92))
        neighbors.append({"title":r['title'],"genre":", ".join(r['genre']),
                          "rating":r['rating'],"episodes":r['episodes'],
                          "poster":poster(r['title']),
                          "dist":round(d, 3),
                          "bar_pct": bar,
                          "pred":le.inverse_transform([y[ni]])[0]})
        if len(neighbors) >= k:
            break

    sr = df.iloc[idx]
    sd = {"title":sel,"poster":poster(sel),"genre":", ".join(sr['genre']),
          "rating":sr['rating'],"pred":le.inverse_transform([y[idx]])[0]}

    body = render_template_string(CLASS_T, titles=ALL_TITLES, sel=sel, k=k, sd=sd, neighbors=neighbors)
    return page("Classification", "klass", body)


# ══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Animatrix — Anime Recommendation System")
    print("  Open: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)