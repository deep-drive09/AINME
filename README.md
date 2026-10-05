# ⛩ ANIMATRIX — Anime Recommendation System

A data mining web app that recommends anime using **association rules**, **K-Means clustering** and **KNN classification**, built as a single-file Flask application.

> Data Mining project

---

## Features

### 🏠 Dashboard: genre-based recommendations
- Pick one or more genres from clickable chips
- Get the top 5 anime that match the most selected genres, ranked by rating
- Each result shows the poster, rating, genres and episode count, with quick links to Association and Classification

### 🔗 Association mining
- "People who watched **X** also watched…" based on 25 watch-history sessions
- Calculates **support**, **confidence** and **lift** for each pair of anime
- If fewer than 4 associations are found, it fills in anime with overlapping genres

### 🌐 Genre clustering
- One-hot encodes genres with `MultiLabelBinarizer`
- Groups the 30 anime into **5 clusters** with **K-Means**
- Reduces the data to 2D with **PCA** and plots it as an interactive scatter chart (Chart.js)
- Click a cluster in the legend to jump to its anime

### 🤖 KNN classification
- Features: one-hot genres + rating + episodes + popularity
- Finds the **K nearest anime** using Euclidean distance (K = 3, 5, 7 or 10)
- Shows each neighbor's distance, with a similarity bar

---

## Data Mining Techniques

| Technique | Library | Used for |
|---|---|---|
| Association rules (support, confidence, lift) | Custom Python | "Also watched" recommendations |
| K-Means clustering (k = 5) | scikit-learn | Grouping anime by genre |
| PCA (2 components) | scikit-learn | 2D visualisation of clusters |
| K-Nearest Neighbors | scikit-learn | Finding similar anime |
| One-hot encoding | `MultiLabelBinarizer` | Turning genre lists into numbers |
| Label encoding | `LabelEncoder` | Class labels for KNN |

### Association formulas
```
Support(A)        = sessions containing A / total sessions
Confidence(A → B) = Support(A ∪ B) / Support(A)
Lift(A → B)       = Support(A ∪ B) / (Support(A) × Support(B))
```
Lift > 1 means people who watch A are more likely than average to watch B.

---

## Tech Stack

| Part | Technology |
|---|---|
| Backend | Python, Flask |
| Data | pandas, NumPy |
| Machine learning | scikit-learn |
| Charts | Chart.js 4 (CDN) |
| Fonts | Bebas Neue, Outfit (Google Fonts) |
| Posters | MyAnimeList CDN images |

---

## Dataset

Built into the code:
- **30 anime**, each with title, genres, rating, episodes and popularity rank
- **25 watch-history sessions**, the "transactions" used for association mining

---

## Pages

| Route | Page |
|---|---|
| `/` | Dashboard and genre selection |
| `/recommend` (POST) | Genre-based results |
| `/association?anime=<title>` | Association mining |
| `/clustering` | K-Means clusters and PCA plot |
| `/classification?anime=<title>&k=<k>` | KNN nearest neighbors |

---

## How to Run

```bash
pip install flask pandas numpy scikit-learn
python anime_recommender.py
```
Open **http://127.0.0.1:5000** in your browser.

An internet connection is needed for posters, fonts and Chart.js.

---

## Project Structure

```
anime_recommender.py   # everything: data, CSS, HTML templates, routes and ML logic
```

---

## Limitations

- Small hardcoded dataset (30 anime, 25 sessions), so results are illustrative, not real-world accurate
- Watch histories are sample data, not real user data
- No user accounts or saved preferences

## Future Scope

- Use a real dataset, such as the Kaggle MyAnimeList data
- Use the Apriori or FP-Growth algorithm from `mlxtend`
- Add collaborative filtering based on user ratings
- Scale features before KNN for fairer distance comparison
- Save user ratings and watch history in a database
