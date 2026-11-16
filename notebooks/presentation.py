import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Hierarchical Bayesian Models and Partial Pooling II")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import altair as alt
    from scipy import stats
    from scipy.special import betaln
    from scipy.optimize import minimize

    # colour roles used in every chart
    C_NO, C_PART, C_POOL, C_MUT, C_INK = "#eb6834", "#2a78d6", "#1baf7a", "#52514e", "#0b0b0b"
    C_EXTRA = ["#1baf7a", "#6250d6", "#eda100", "#e87ba4"]

    def md_table(df):
        """Render a small polars DataFrame as a clean markdown table."""
        head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
        body = "".join("| " + " | ".join(str(v) for v in row) + " |\n" for row in df.iter_rows())
        return mo.md(head + body)
    return (
        C_EXTRA,
        C_INK,
        C_MUT,
        C_NO,
        C_PART,
        C_POOL,
        alt,
        betaln,
        minimize,
        mo,
        np,
        pl,
        md_table,
        stats,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hierarchical Bayesian Models and Partial Pooling II
    ### Shrinkage Estimation and Bayesian Multilevel Regression

    **Kutay Doğan · Yağmur Deniz Gülbaz**

    Running example: a movie platform records whether each viewer **liked** or **disliked** a film.

    - Shrinkage in a hierarchical beta–binomial model
    - Learning the amount of shrinkage with hyperpriors
    - Hierarchical Bayesian (multilevel) logistic regression

    *Move the sliders: every chart is recomputed live.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. Introduction

    Grouped data: groups differ, but they are related.

    - Ratings → movies, patients → hospitals, measurements → individuals
    - Treat groups as **unrelated** → information is wasted
    - Treat groups as **identical** → real differences are ignored
    - Movie A: 9 / 10 likes (90%) · Movie B: 800 / 1000 likes (80%)
    - A looks better, but one extra dislike drops A to about 82%
    - Ranking by raw rates favours movies with **few votes**
    """)
    return


@app.cell
def _(mo):
    intro_n = mo.ui.slider(
        steps=[3, 5, 10, 20, 50, 100, 200, 500, 1000], value=10, label="Votes $n$", show_value=True
    )
    intro_p = mo.ui.slider(0.05, 0.95, step=0.05, value=0.9, label="Observed approval rate", show_value=True)
    return intro_n, intro_p


@app.cell(hide_code=True)
def _(C_MUT, C_NO, alt, intro_n, intro_p, mo, np, pl):
    _ns = np.unique(np.round(np.logspace(np.log10(3), 3, 120)).astype(int))
    _p = intro_p.value
    _df = pl.DataFrame({"n": _ns, "half_width": 100 * 1.96 * np.sqrt(_p * (1 - _p) / _ns)})
    _cur = pl.DataFrame({"n": [intro_n.value], "half_width": [100 * 1.96 * np.sqrt(_p * (1 - _p) / intro_n.value)]})
    _line = alt.Chart(_df).mark_line(color=C_MUT, strokeWidth=2).encode(
        x=alt.X("n:Q", scale=alt.Scale(type="log"), title="Number of votes n (log scale)"),
        y=alt.Y("half_width:Q", title="± margin of the raw rate (percentage points)"),
    )
    _pt = alt.Chart(_cur).mark_point(color=C_NO, size=120, filled=True).encode(
        x="n:Q", y="half_width:Q", tooltip=["n", alt.Tooltip("half_width:Q", format=".1f")]
    )
    _likes = round(_p * intro_n.value)
    _after = 100 * _likes / (intro_n.value + 1)
    mo.vstack([
        mo.hstack([intro_n, intro_p], justify="start", gap=2),
        mo.hstack([
            (_line + _pt).properties(width=380, height=230, title="Few votes → wide uncertainty"),
            mo.md(f"""
    - 95% margin: **±{_cur['half_width'][0]:.1f} points**
    - One extra dislike: {100*_likes/intro_n.value:.0f}% → **{_after:.1f}%**
    - Margin shrinks like $1/\\sqrt{{n}}$
    """),
        ], align="center", justify="start", gap=2),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2. A Hierarchical Beta–Binomial Model

    ### 2.1 Two levels of variation

    Each movie has its own like probability; all probabilities come from one shared distribution.

    $$
    Y_j \mid \theta_j \sim \operatorname{Binomial}(n_j,\theta_j),
    \qquad
    \theta_j \mid \alpha,\beta \overset{iid}{\sim} \operatorname{Beta}(\alpha,\beta)
    $$

    - $Y_j$: number of likes of movie $j$; $n_j$: number of votes
    - $\theta_j$: true probability that a viewer likes movie $j$
    - $\alpha,\beta>0$: shared shape parameters → link all movies
    - Assumptions: votes independent given $\theta_j$; same probability for every vote of movie $j$
    - Beta: lives on $(0,1)$ and is conjugate to the binomial

    $$
    m=\frac{\alpha}{\alpha+\beta},\qquad \kappa=\alpha+\beta,\qquad
    \operatorname{Var}(\theta_j)=\frac{m(1-m)}{\kappa+1}
    $$

    - $m$: numerator $\alpha$ = "pseudo-likes", denominator $\alpha+\beta$ = "pseudo-votes" → average approval
    - $\kappa$: **concentration**, total pseudo-votes
    - Variance: numerator $m(1-m)$ = largest possible spread at mean $m$; denominator $\kappa+1$ → larger $\kappa$, smaller spread
    """)
    return


@app.cell
def _(mo):
    beta_m = mo.ui.slider(0.05, 0.95, step=0.05, value=0.7, label="Population mean $m$", show_value=True)
    beta_k = mo.ui.slider(
        steps=[0.5, 1, 2, 4, 6, 10, 20, 50, 100, 200], value=10, label="Concentration $\\kappa$", show_value=True
    )
    return beta_k, beta_m


@app.cell(hide_code=True)
def _(C_MUT, C_PART, alt, beta_k, beta_m, mo, np, pl, stats):
    _a, _b = beta_m.value * beta_k.value, (1 - beta_m.value) * beta_k.value
    _x = np.linspace(0.001, 0.999, 400)
    _dens = pl.DataFrame({"theta": _x, "density": stats.beta.pdf(_x, _a, _b)})
    _dens_chart = alt.Chart(_dens).mark_area(color=C_PART, opacity=0.35, line={"color": C_PART}).encode(
        x=alt.X("theta:Q", title="Movie approval probability θ"),
        y=alt.Y("density:Q", title="Density", scale=alt.Scale(domain=[0, min(_dens["density"].max() * 1.05, 15)], clamp=True)),
    ).properties(width=340, height=220, title=f"Beta({_a:.2f}, {_b:.2f})")

    _ks = np.logspace(-0.5, 2.5, 100)
    _corr = pl.DataFrame({"kappa": _ks, "corr": 1 / (_ks + 1)})
    _corr_chart = (
        alt.Chart(_corr).mark_line(color=C_MUT).encode(
            x=alt.X("kappa:Q", scale=alt.Scale(type="log"), title="κ (log scale)"),
            y=alt.Y("corr:Q", title="Corr. of two votes, same movie"),
        )
        + alt.Chart(pl.DataFrame({"kappa": [beta_k.value], "corr": [1 / (beta_k.value + 1)]}))
        .mark_point(color=C_PART, size=120, filled=True).encode(x="kappa:Q", y="corr:Q")
    ).properties(width=300, height=220, title="Induced dependence = 1/(κ+1)")

    _sd = np.sqrt(beta_m.value * (1 - beta_m.value) / (beta_k.value + 1))
    mo.vstack([
        mo.hstack([beta_m, beta_k], justify="start", gap=2),
        mo.hstack([_dens_chart, _corr_chart], justify="start", gap=2),
        mo.md(f"α = **{_a:.2f}**, β = **{_b:.2f}**, SD(θ) = **{_sd:.3f}**, Corr = **{1/(beta_k.value+1):.3f}**  \n"
              "Increase κ → movies cluster around m → votes for the same movie are less correlated."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2 Exchangeability and induced dependence

    Given $\alpha,\beta$, movie labels carry no information (**conditional exchangeability**).

    $$
    Z_{ij}\mid\theta_j\sim\operatorname{Bernoulli}(\theta_j),
    \qquad
    \theta_j\mid Z_{1j}=1\sim\operatorname{Beta}(\alpha+1,\beta)
    $$

    - $Z_{ij}$: vote of viewer $i$ for movie $j$ (1 = like); $Y_j=\sum_i Z_{ij}$
    - One like adds 1 to $\alpha$ → next vote's predicted like probability rises from $\frac{\alpha}{\alpha+\beta}$ to $\frac{\alpha+1}{\alpha+\beta+1}$

    $$
    \operatorname{Cov}(Z_{ij},Z_{i'j})=\operatorname{Var}(\theta_j)=\frac{m(1-m)}{\kappa+1},
    \qquad
    \operatorname{Corr}(Z_{ij},Z_{i'j})=\frac{m(1-m)/(\kappa+1)}{m(1-m)}=\frac{1}{\kappa+1}
    $$

    - Covariance of two votes = variance of their shared unknown $\theta_j$
    - Correlation: numerator = covariance, denominator = variance of one vote $m(1-m)$
    - Votes are independent **given** $\theta_j$, dependent once $\theta_j$ is unknown (right chart above)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. Pooling Strategies

    Shrinkage = compromise between two simpler strategies.

    $$
    \text{Complete pooling: } \hat\theta_{\text{pool}}=\frac{\sum_j y_j}{\sum_j n_j}
    \qquad
    \text{No pooling: } \hat p_j=\frac{y_j}{n_j}
    $$

    - Complete pooling: numerator = all likes, denominator = all votes → one rate for every movie
    - No pooling: each movie alone → unbiased but noisy for small $n_j$ (1/1 → 100%)
    - Partial pooling: own $\theta_j$, shared $\operatorname{Beta}(\alpha,\beta)$
    - $\kappa\to 0$: no pooling · $\kappa\to\infty$: every movie gets $m$ (complete pooling)
    - Estimating $\alpha,\beta$ → **the data choose** the point in between
    """)
    return


@app.cell
def _(mo):
    pool_logk = mo.ui.slider(-1, 4, step=0.1, value=1, label="$\\log_{10}\\kappa$", show_value=True)
    pool_m = mo.ui.slider(0.5, 0.9, step=0.05, value=0.7, label="Population mean $m$", show_value=True)
    return pool_logk, pool_m


@app.cell(hide_code=True)
def _(C_EXTRA, C_MUT, alt, mo, np, pl, pool_logk, pool_m):
    _movies = {"A": (9, 10), "B": (800, 1000), "C": (35, 50), "D": (3, 5)}
    _lk = np.linspace(-1, 4, 200)
    _rows = []
    for _name, (_y, _n) in _movies.items():
        _w = _n / (_n + 10**_lk)
        for _l, _ww in zip(_lk, _w):
            _rows.append((_name, 10**_l, _ww * _y / _n + (1 - _ww) * pool_m.value))
    _df = pl.DataFrame(_rows, schema=["movie", "kappa", "estimate"], orient="row")
    _k = 10**pool_logk.value
    _cur = pl.DataFrame(
        [(nm, _k, (_n / (_n + _k)) * _y / _n + (_k / (_n + _k)) * pool_m.value) for nm, (_y, _n) in _movies.items()],
        schema=["movie", "kappa", "estimate"], orient="row",
    )
    _col = alt.Color("movie:N", scale=alt.Scale(range=C_EXTRA), title="Movie")
    _chart = (
        alt.Chart(_df).mark_line(strokeWidth=2).encode(
            x=alt.X("kappa:Q", scale=alt.Scale(type="log"), title="κ (log scale): no pooling ← → complete pooling"),
            y=alt.Y("estimate:Q", title="Estimated approval", scale=alt.Scale(domain=[0.55, 0.95])),
            color=_col,
        )
        + alt.Chart(pl.DataFrame({"kappa": [_k]})).mark_rule(color=C_MUT, strokeDash=[4, 3]).encode(x="kappa:Q")
        + alt.Chart(_cur).mark_point(size=110, filled=True).encode(
            x="kappa:Q", y="estimate:Q", color=_col, tooltip=["movie", alt.Tooltip("estimate:Q", format=".3f")]
        )
    ).properties(width=520, height=260, title="A–D: from their own rates (left) to m (right)")
    _tbl = _cur.with_columns((pl.col("estimate") * 100).round(1).alias("estimate %")).select("movie", "estimate %")
    mo.vstack([mo.hstack([pool_logk, pool_m], justify="start", gap=2), mo.hstack([_chart, md_table(_tbl)], align="center", justify="start", gap=2)], justify="start", gap=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. Shrinkage Estimation in the Beta–Binomial Model

    ### 4.1 Conditional posterior

    Treat $\alpha,\beta$ as fixed for now.

    $$
    \theta_j\mid y_j,n_j,\alpha,\beta\sim\operatorname{Beta}(\alpha+y_j,\ \beta+n_j-y_j)
    $$

    - Prior pseudo-likes $\alpha$ + observed likes $y_j$
    - Prior pseudo-dislikes $\beta$ + observed dislikes $n_j-y_j$

    $$
    \mathbb E(\theta_j\mid y_j)=\frac{y_j+\alpha}{n_j+\alpha+\beta}
    = w_j\,\hat p_j+(1-w_j)\,m,
    \qquad
    w_j=\frac{n_j}{n_j+\kappa}
    $$

    - Numerator $y_j+\alpha$: real likes + pseudo-likes; denominator $n_j+\alpha+\beta$: real votes + pseudo-votes
    - $\hat p_j=y_j/n_j$: raw rate · $m$: population mean
    - $w_j$: weight on the movie's own data (votes ÷ (votes + κ))
    - $B_j=1-w_j=\kappa/(n_j+\kappa)$: **shrinkage factor**
    - Movement is **toward $m$**: high rates go down, low rates go up
    """)
    return


@app.cell
def _(mo):
    post_a = mo.ui.slider(0.5, 30, step=0.5, value=7, label="Prior $\\alpha$", show_value=True)
    post_b = mo.ui.slider(0.5, 30, step=0.5, value=3, label="Prior $\\beta$", show_value=True)
    post_n = mo.ui.slider(1, 200, step=1, value=10, label="Votes $n_j$", show_value=True)
    post_rate = mo.ui.slider(0.0, 1.0, step=0.05, value=0.9, label="Raw rate $\\hat p_j$", show_value=True)
    return post_a, post_b, post_n, post_rate


@app.cell(hide_code=True)
def _(C_MUT, C_NO, C_PART, alt, mo, np, pl, post_a, post_b, post_n, post_rate, stats):
    _a, _b, _n = post_a.value, post_b.value, post_n.value
    _y = int(round(post_rate.value * _n))
    _x = np.linspace(0.001, 0.999, 400)
    _lik = stats.beta.pdf(_x, _y + 1, _n - _y + 1)
    _df = pl.concat([
        pl.DataFrame({"theta": _x, "density": stats.beta.pdf(_x, _a, _b), "curve": "Prior Beta(α, β)"}),
        pl.DataFrame({"theta": _x, "density": _lik, "curve": "Likelihood (scaled)"}),
        pl.DataFrame({"theta": _x, "density": stats.beta.pdf(_x, _a + _y, _b + _n - _y), "curve": "Posterior"}),
    ])
    _chart = alt.Chart(_df).mark_line(strokeWidth=2).encode(
        x=alt.X("theta:Q", title="θ"),
        y=alt.Y("density:Q", title="Density"),
        color=alt.Color("curve:N", scale=alt.Scale(domain=["Prior Beta(α, β)", "Likelihood (scaled)", "Posterior"],
                                                   range=[C_MUT, C_NO, C_PART]), title=None),
        strokeDash=alt.condition(alt.datum.curve == "Prior Beta(α, β)", alt.value([4, 3]), alt.value([1, 0])),
    ).properties(width=400, height=240, title="Posterior sits between prior and data")
    _k = _a + _b
    _w = _n / (_n + _k)
    _m = _a / _k
    mo.vstack([
        mo.hstack([post_a, post_b], justify="start", gap=2),
        mo.hstack([post_n, post_rate], justify="start", gap=2),
        mo.hstack([_chart, mo.md(f"""
    - Data: **{_y} / {_n}** likes → $\\hat p$ = {_y/_n:.3f}
    - $m$ = {_m:.3f}, κ = {_k:.1f}
    - $w_j$ = {_n} / ({_n} + {_k:.1f}) = **{_w:.3f}**
    - Posterior mean = {_w:.3f}·{_y/_n:.3f} + {1-_w:.3f}·{_m:.3f} = **{(_y+_a)/(_n+_k):.3f}**
    - Raise $n_j$ → posterior follows the data
    - Raise α+β → posterior follows the prior
    """)], align="center", justify="start", gap=2),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2 What controls the amount of shrinkage?

    $$
    \operatorname{Var}(\theta_j\mid y_j)=\frac{a_j b_j}{(a_j+b_j)^2(a_j+b_j+1)},
    \qquad a_j=\alpha+y_j,\ b_j=\beta+n_j-y_j
    $$

    - More votes $n_j$ → $w_j\to 1$ → own data dominate
    - Larger $\kappa$ → movies assumed similar → stronger pooling
    - $\kappa$ ≈ "prior effective sample size" (pseudo-counts, not real votes)
    - Posterior variance: numerator $a_jb_j$, denominator grows with total counts → shrinks as votes rise
    - Shrinkage = small **bias** toward $m$ for a large drop in **variance** → lower total error (James–Stein)
    """)
    return


@app.cell
def _(mo):
    bv_k = mo.ui.slider(steps=[1, 2, 5, 10, 20, 50, 100, 200], value=10, label="True $\\kappa$", show_value=True)
    bv_n = mo.ui.slider(steps=[1, 3, 5, 10, 20, 50, 100, 500], value=10, label="Votes per movie $n$", show_value=True)
    return bv_k, bv_n


@app.cell(hide_code=True)
def _(C_MUT, C_NO, C_PART, C_POOL, alt, bv_k, bv_n, mo, np, pl):
    # expected squared error of w*p_hat + (1-w)*m when theta ~ Beta(m, kappa_true), for every assumed kappa
    _m, _kt, _n = 0.7, bv_k.value, bv_n.value
    _V = _m * (1 - _m) / (_kt + 1)                 # between-movie variance
    _A = _m * (1 - _m) * _kt / ((_kt + 1) * _n)    # average sampling variance of p_hat
    _ka = np.logspace(-2, 4, 300)
    _w = _n / (_n + _ka)
    _mse = _w**2 * _A + (1 - _w) ** 2 * _V
    _df = pl.DataFrame({"kappa": _ka, "mse": _mse})
    _best = _n / (_n + _kt)
    _chart = (
        alt.Chart(_df).mark_line(color=C_PART, strokeWidth=2.2).encode(
            x=alt.X("kappa:Q", scale=alt.Scale(type="log"), title="κ used for shrinkage (log scale)"),
            y=alt.Y("mse:Q", title="Expected squared error"),
        )
        + alt.Chart(pl.DataFrame({"y": [_A], "lab": ["No pooling"]})).mark_rule(color=C_NO, strokeDash=[4, 3]).encode(y="y:Q")
        + alt.Chart(pl.DataFrame({"y": [_V], "lab": ["Complete pooling"]})).mark_rule(color=C_POOL, strokeDash=[6, 3]).encode(y="y:Q")
        + alt.Chart(pl.DataFrame({"kappa": [_kt]})).mark_rule(color=C_MUT).encode(x="kappa:Q")
    ).properties(width=430, height=240, title="Bias–variance trade-off (m = 0.7)")
    mo.vstack([
        mo.hstack([bv_k, bv_n], justify="start", gap=2),
        mo.hstack([_chart, mo.md(f"""
    - <span style="color:{C_NO}">**dashed orange**</span>: no pooling error = {_A:.4f}
    - <span style="color:{C_POOL}">**dashed green**</span>: complete pooling error = {_V:.4f}
    - Best κ = **true κ** (grey line), error = {_mse.min():.4f}
    - Gain over no pooling: **{100*(1-_mse.min()/_A):.0f}%**
    - Few votes → big gain · many votes → almost none
    """)], align="center", justify="start", gap=2),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.3 A numerical movie example

    Four hand-picked hypothetical movies; illustrative prior $\operatorname{Beta}(7,3)$ → $m=0.7$, $\kappa=10$.

    - A: $w_A=10/(10+10)=0.5$ → $0.5(0.90)+0.5(0.70)=0.80$
    - B: $w_B\approx0.99$ → barely moves
    - C: raw rate = $m$ → no movement
    - D: pulled **up** from 0.60 to $2/3$
    - Complete pooling: same 79.5% for all, with a misleadingly small ±1.2
    """)
    return


@app.cell
def _(mo):
    tab_a = mo.ui.slider(0.5, 50, step=0.5, value=7, label="$\\alpha$", show_value=True)
    tab_b = mo.ui.slider(0.5, 50, step=0.5, value=3, label="$\\beta$", show_value=True)
    return tab_a, tab_b


@app.cell(hide_code=True)
def _(C_EXTRA, C_MUT, alt, mo, np, pl, tab_a, tab_b):
    _y = np.array([9, 800, 35, 3]); _n = np.array([10, 1000, 50, 5])
    _a, _b = tab_a.value, tab_b.value
    _pp = _y.sum() / _n.sum(); _sp = np.sqrt(_pp * (1 - _pp) / _n.sum())
    _raw = _y / _n; _sr = np.sqrt(_raw * (1 - _raw) / _n)
    _A, _B = _a + _y, _b + _n - _y
    _pm = _A / (_A + _B); _ps = np.sqrt(_A * _B / ((_A + _B) ** 2 * (_A + _B + 1)))
    _tbl = pl.DataFrame({
        "Movie": list("ABCD"), "y": _y, "n": _n,
        "w_j": np.round(_n / (_n + _a + _b), 3),
        "Complete %": [f"{100*_pp:.1f} ({100*_sp:.1f})"] * 4,
        "No pooling %": [f"{100*r:.1f} ({100*s:.1f})" for r, s in zip(_raw, _sr)],
        "Partial %": [f"{100*p:.1f} ({100*s:.1f})" for p, s in zip(_pm, _ps)],
    })
    _arrows = pl.DataFrame({"movie": list("ABCD"), "raw": _raw, "post": _pm})
    _chart = (
        alt.Chart(_arrows).mark_rule(color=C_MUT, strokeWidth=1.5).encode(
            y=alt.Y("movie:N", title=None), x=alt.X("raw:Q", title="Approval", scale=alt.Scale(domain=[0.5, 1])), x2="post:Q")
        + alt.Chart(_arrows).mark_point(shape="circle", size=90, filled=True, color="#eb6834").encode(y="movie:N", x="raw:Q")
        + alt.Chart(_arrows).mark_point(shape="triangle-right", size=110, filled=True, color="#2a78d6").encode(y="movie:N", x="post:Q")
        + alt.Chart(pl.DataFrame({"m": [_a / (_a + _b)]})).mark_rule(strokeDash=[2, 2], color=C_MUT).encode(x="m:Q")
    ).properties(width=380, height=170, title="Raw rate (orange) → posterior mean (blue)")
    mo.vstack([mo.hstack([tab_a, tab_b], justify="start", gap=2), md_table(_tbl), _chart,
               mo.md("Values in %, SE / posterior SD in parentheses. Default α=7, β=3 reproduces **Table 1** of the report.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 5. Hyperpriors

    ### 5.1 Why estimate the shared parameters?

    Fixing $\operatorname{Beta}(7,3)$ imposes $m=0.7$, $\kappa=10$ without evidence → give $\alpha,\beta$ their own priors.

    $$
    Y_j\mid\theta_j\sim\operatorname{Binomial}(n_j,\theta_j),\quad
    \theta_j\mid\alpha,\beta\sim\operatorname{Beta}(\alpha,\beta),\quad
    \alpha\sim\operatorname{Gamma}(a_\alpha,b_\alpha),\quad
    \beta\sim\operatorname{Gamma}(a_\beta,b_\beta)
    $$

    - Gamma: positive values only, as $\alpha,\beta$ require
    - Shape–rate notation: mean = shape / rate
    - Our choice $\operatorname{Gamma}(2,1)$ is **not neutral**: prior means 2 → $m\approx0.5$, $\kappa\approx4$
    - Small $\kappa$ = large differences between movies = weak shrinkage
    - Similar movie rates → data favour large $\kappa$ (strong shrinkage); very different rates → small $\kappa$
    """)
    return


@app.cell
def _(mo):
    hp_shape = mo.ui.slider(0.5, 10, step=0.5, value=2, label="Gamma shape", show_value=True)
    hp_rate = mo.ui.slider(0.05, 2, step=0.05, value=1, label="Gamma rate", show_value=True)
    return hp_rate, hp_shape


@app.cell(hide_code=True)
def _(C_PART, C_POOL, alt, hp_rate, hp_shape, mo, np, pl):
    _rng = np.random.default_rng(1)
    _al = _rng.gamma(hp_shape.value, 1 / hp_rate.value, 20000)
    _be = _rng.gamma(hp_shape.value, 1 / hp_rate.value, 20000)
    _c1, _e1 = np.histogram(_al / (_al + _be), bins=40, range=(0, 1))
    _c2, _e2 = np.histogram(np.log10(_al + _be), bins=40, range=(-1.5, 2.5))
    _d1 = pl.DataFrame({"lo": _e1[:-1], "hi": _e1[1:], "count": _c1})
    _d2 = pl.DataFrame({"lo": _e2[:-1], "hi": _e2[1:], "count": _c2})
    _h1 = alt.Chart(_d1).mark_rect(color=C_PART, opacity=0.8).encode(
        x=alt.X("lo:Q", title="Implied prior on m"), x2="hi:Q", y=alt.Y("count:Q", title="Draws"))
    _h2 = alt.Chart(_d2).mark_rect(color=C_POOL, opacity=0.8).encode(
        x=alt.X("lo:Q", title="Implied prior on log10 κ"), x2="hi:Q", y=alt.Y("count:Q", title=None))
    mo.vstack([
        mo.hstack([hp_shape, hp_rate], justify="start", gap=2),
        mo.hstack([_h1.properties(width=300, height=200), _h2.properties(width=300, height=200)], justify="start", gap=2),
        mo.md(f"Prior mean of κ = 2 × shape / rate = **{2*hp_shape.value/hp_rate.value:.1f}**, "
              f"median κ = **{np.median(_al+_be):.1f}** → raise the rate to expect *more* variation between movies."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.2 Joint and marginal posteriors

    $$
    p(\alpha,\beta\mid\mathbf y,\mathbf n)\ \propto\ p(\alpha)\,p(\beta)\prod_{j=1}^{J}
    \frac{B(\alpha+y_j,\ \beta+n_j-y_j)}{B(\alpha,\beta)}
    $$

    - $B(\cdot,\cdot)$: Beta function; each $\theta_j$ integrated out analytically
    - Numerator: how well $(\alpha,\beta)$ explain movie $j$'s likes and dislikes; denominator: normalises the Beta prior
    - Only **two** unknowns left → evaluate on a grid

    $$
    \mathbb E(\theta_j\mid\mathbf y,\mathbf n)=\mathbb E_{\alpha,\beta\mid\mathbf y,\mathbf n}\!\left[\frac{y_j+\alpha}{n_j+\alpha+\beta}\right]
    $$

    - Fully Bayesian shrinkage: average the conditional estimate over the posterior of $(\alpha,\beta)$
    - The weight $n_j/(n_j+\alpha+\beta)$ now varies across posterior draws

    ### 5.3 Empirical Bayes vs full Bayes

    - Empirical Bayes: plug in the best $(\hat\alpha,\hat\beta)$ → ignores their uncertainty
    - Full Bayes: averages over $(\alpha,\beta)$ → matters with few movies or few votes
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.4 Shrinkage across many movies

    Synthetic data: $\theta_j\sim\operatorname{Beta}(m\kappa,(1-m)\kappa)$, $n_j$ log-uniform on $[3,N_{\max}]$, $y_j\sim\operatorname{Binomial}(n_j,\theta_j)$.
    Defaults reproduce **Figure 1** (seed 2026, Beta(7,3), Gamma(2,1) hyperpriors).

    - Few votes → long arrows · hundreds of votes → no movement
    - Estimates move toward **$m$** (dotted), not the complete-pooling rate (dashed)
    - Complete pooling counts every **vote**; $m$ counts every **movie** once
    """)
    return


@app.cell
def _(mo):
    sim_m = mo.ui.slider(0.3, 0.9, step=0.05, value=0.7, label="True $m$", show_value=True)
    sim_k = mo.ui.slider(steps=[2, 5, 10, 20, 50, 100], value=10, label="True $\\kappa$", show_value=True)
    sim_j = mo.ui.slider(10, 100, step=5, value=30, label="Movies drawn", show_value=True)
    sim_nmax = mo.ui.slider(steps=[50, 200, 500, 1500, 5000], value=1500, label="Max votes", show_value=True)
    sim_shape = mo.ui.slider(0.5, 10, step=0.5, value=2, label="Hyperprior shape", show_value=True)
    sim_rate = mo.ui.slider(0.05, 2, step=0.05, value=1, label="Hyperprior rate", show_value=True)
    sim_seed = mo.ui.number(start=0, stop=10_000, value=2026, label="Seed")
    return sim_j, sim_k, sim_m, sim_nmax, sim_rate, sim_seed, sim_shape


@app.cell
def _(betaln, np, sim_j, sim_k, sim_m, sim_nmax, sim_rate, sim_seed, sim_shape, stats):
    # identical random-number sequence to shrinkage_plot.py
    _rng = np.random.default_rng(int(sim_seed.value))
    sim_n = np.unique(np.round(np.exp(_rng.uniform(np.log(3), np.log(sim_nmax.value), sim_j.value))).astype(int))
    _a_true = round(sim_m.value * sim_k.value, 10)
    _b_true = round((1 - sim_m.value) * sim_k.value, 10)
    sim_theta = _rng.beta(_a_true, _b_true, len(sim_n))
    sim_y = _rng.binomial(sim_n, sim_theta)

    _g = np.exp(np.linspace(np.log(0.05), np.log(200), 500))
    _A, _B = np.meshgrid(_g, _g, indexing="ij")
    _lp = (stats.gamma.logpdf(_A, sim_shape.value, scale=1 / sim_rate.value)
           + stats.gamma.logpdf(_B, sim_shape.value, scale=1 / sim_rate.value) + np.log(_A) + np.log(_B))
    for _yj, _nj in zip(sim_y, sim_n):
        _lp += betaln(_A + _yj, _B + _nj - _yj) - betaln(_A, _B)
    _w = np.exp(_lp - _lp.max()); _w /= _w.sum()
    sim_post = np.array([(_w * (_yj + _A) / (_nj + _A + _B)).sum() for _yj, _nj in zip(sim_y, sim_n)])
    sim_raw = sim_y / sim_n
    sim_pool = sim_y.sum() / sim_n.sum()
    sim_mhat = float((_w * _A / (_A + _B)).sum())
    sim_khat = float((_w * (_A + _B)).sum())
    return sim_khat, sim_mhat, sim_n, sim_pool, sim_post, sim_raw, sim_theta, sim_y


@app.cell(hide_code=True)
def _(
    C_MUT, C_NO, C_PART, C_POOL, alt, mo, np, pl, sim_j, sim_k, sim_khat, sim_m, sim_mhat, sim_n, sim_nmax,
    sim_pool, sim_post, sim_rate, sim_raw, sim_seed, sim_shape, sim_theta, sim_y,
):
    _df = pl.DataFrame({"n": sim_n, "y": sim_y, "raw": sim_raw, "post": sim_post, "true": sim_theta,
                        "log10n": np.log10(sim_n)})
    _long = pl.concat([
        _df.select("n", pl.col("raw").alias("estimate"), pl.lit("No pooling (y/n)").alias("method")),
        _df.select("n", pl.col("post").alias("estimate"), pl.lit("Partial pooling").alias("method")),
    ])
    _col = alt.Color("method:N", scale=alt.Scale(domain=["No pooling (y/n)", "Partial pooling"], range=[C_NO, C_PART]),
                     title=None, legend=alt.Legend(orient="bottom"))
    _lines = (
        alt.Chart(pl.DataFrame({"y": [sim_pool]})).mark_rule(color=C_POOL, strokeWidth=1.8, strokeDash=[6, 3]).encode(y="y:Q")
        + alt.Chart(pl.DataFrame({"y": [sim_mhat]})).mark_rule(color=C_MUT, strokeWidth=1.5, strokeDash=[2, 2]).encode(y="y:Q")
    )
    _pa = (
        alt.Chart(_df).mark_rule(color=C_MUT, opacity=0.7).encode(
            x=alt.X("n:Q", scale=alt.Scale(type="log"), title="Votes n_j (log scale)"), y="raw:Q", y2="post:Q")
        + alt.Chart(_long).mark_point(filled=True, size=55, opacity=0.9).encode(
            x="n:Q", y=alt.Y("estimate:Q", title="Estimated approval", scale=alt.Scale(zero=False)), color=_col,
            tooltip=["n", alt.Tooltip("estimate:Q", format=".3f"), "method"])
        + _lines
    ).properties(width=340, height=270, title="(a) Shrinkage depends on n_j")
    _lo = float(min(sim_raw.min(), sim_post.min())) - 0.03
    _hi = float(max(sim_raw.max(), sim_post.max())) + 0.03
    _pb = (
        alt.Chart(pl.DataFrame({"x": [_lo, _hi], "y": [_lo, _hi]})).mark_line(color=C_NO).encode(x="x:Q", y="y:Q")
        + alt.Chart(_df).mark_point(filled=True, size=60, stroke=C_PART).encode(
            x=alt.X("raw:Q", title="Raw rate y_j/n_j", scale=alt.Scale(domain=[_lo, _hi])),
            y=alt.Y("post:Q", title="Posterior mean", scale=alt.Scale(domain=[_lo, _hi])),
            color=alt.Color("log10n:Q", scale=alt.Scale(scheme="blues", domain=[0, 3.5]), title="log10 n"),
            tooltip=["n", "y", alt.Tooltip("raw:Q", format=".3f"), alt.Tooltip("post:Q", format=".3f")])
        + _lines
    ).properties(width=300, height=270, title="(b) Between the two extremes")
    _mse = lambda e: float(np.mean((e - sim_theta) ** 2))
    _stats = pl.DataFrame({
        "Quantity": ["Movies kept", "Complete-pooling rate", "Posterior mean of m", "Posterior mean of κ",
                     "MSE no pooling", "MSE complete pooling", "MSE partial pooling"],
        "Value": [f"{len(sim_n)}", f"{sim_pool:.3f}", f"{sim_mhat:.3f}", f"{sim_khat:.2f} (true {sim_k.value})",
                  f"{_mse(sim_raw):.4f}", f"{_mse(sim_pool):.4f}", f"{_mse(sim_post):.4f}"],
    })
    mo.vstack([
        mo.hstack([sim_m, sim_k, sim_j], justify="start", gap=2),
        mo.hstack([sim_nmax, sim_shape, sim_rate, sim_seed], justify="start", gap=2),
        alt.hconcat(_pa, _pb).resolve_scale(color="independent"),
        mo.md(f"<span style='color:{C_POOL}'>**- - green dashed**</span>: complete pooling ({sim_pool:.3f}) · "
              f"<span style='color:{C_MUT}'>**··· grey dotted**</span>: estimated m ({sim_mhat:.3f})"),
        mo.hstack([md_table(_stats), mo.md("""
    **Try:**

    - Raise true κ → movies more alike → stronger shrinkage
    - Lower max votes → every movie noisy → partial pooling wins clearly
    - Raise hyperprior shape (e.g. 10) → prior allows large κ → estimate of κ moves toward the truth
    """)], align="start", justify="start", gap=2),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 6. Hierarchical Bayesian Regression

    ### 6.1 Shrinkage in a normal hierarchical model

    $$
    \bar y_j\mid\mu_j\sim\mathcal N\!\left(\mu_j,\frac{\sigma^2}{n_j}\right),\qquad
    \mu_j\mid\mu,\tau\sim\mathcal N(\mu,\tau^2)
    $$

    $$
    \mathbb E(\mu_j\mid\bar y_j)=w_j\bar y_j+(1-w_j)\mu,\qquad
    w_j=\frac{n_j}{n_j+\sigma^2/\tau^2}
    $$

    - $\bar y_j$: group average; $\sigma^2$: within-group variance; $\tau^2$: between-group variance
    - Same form as the beta–binomial weight: $\sigma^2/\tau^2$ plays the role of $\kappa$
    - Numerator $n_j$: the group's own information; denominator adds the "prior sample size" $\sigma^2/\tau^2$
    - $\tau\to0$: complete pooling · $\tau\to\infty$: no pooling
    """)
    return


@app.cell
def _(mo):
    nh_sigma = mo.ui.slider(0.5, 5, step=0.25, value=2, label="Within-group SD $\\sigma$", show_value=True)
    nh_tau = mo.ui.slider(0.05, 3, step=0.05, value=0.5, label="Between-group SD $\\tau$", show_value=True)
    return nh_sigma, nh_tau


@app.cell(hide_code=True)
def _(C_PART, alt, mo, nh_sigma, nh_tau, np, pl):
    _ratio = nh_sigma.value**2 / nh_tau.value**2
    _ns = np.unique(np.round(np.logspace(0, 3, 150)).astype(int))
    _df = pl.DataFrame({"n": _ns, "w": _ns / (_ns + _ratio)})
    _chart = alt.Chart(_df).mark_line(color=C_PART, strokeWidth=2.2).encode(
        x=alt.X("n:Q", scale=alt.Scale(type="log"), title="Observations in the group n_j (log scale)"),
        y=alt.Y("w:Q", title="Weight on own group mean w_j", scale=alt.Scale(domain=[0, 1])),
        tooltip=["n", alt.Tooltip("w:Q", format=".3f")],
    ).properties(width=420, height=230, title="Groups need about σ²/τ² observations to get w = 0.5")
    mo.vstack([mo.hstack([nh_sigma, nh_tau], justify="start", gap=2), mo.hstack([_chart, mo.md(
        f"- σ²/τ² = **{_ratio:.1f}**\n- Raise τ → groups differ more → less shrinkage\n- Raise σ → noisier data → more shrinkage")],
        align="center", justify="start", gap=2)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.2 From movie proportions to logistic regression

    Model individual votes; add a viewer-level predictor $x_{ij}$ (e.g. genre preference, centred).

    $$
    Z_{ij}\sim\operatorname{Bernoulli}(p_{ij}),\qquad
    \operatorname{logit}(p_{ij})=a_j+b\,x_{ij},\qquad
    a_j\sim\mathcal N(\mu_a,\tau_a^2)
    $$

    - $\operatorname{logit}(p)=\log\frac{p}{1-p}$: numerator = probability of a like, denominator = probability of a dislike
    - $a_j$: movie $j$'s baseline log-odds · $b$: shared effect of the predictor
    - $\mu_a$: average baseline · $\tau_a$: how much baselines vary → controls shrinkage of $a_j$
    - Intercepts are normal on the **log-odds** scale → weighted-average formula holds only approximately
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.3 Movie-level predictors and varying slopes

    $$
    a_j\sim\mathcal N(\gamma_0+\gamma_1 d_j+\gamma_2 g_j,\ \tau_a^2)
    $$

    - $d_j$: standardised duration, $g_j$: genre indicator → shrink toward a **conditional** mean
    - Reviewer effect $r_u\sim\mathcal N(0,\tau_r^2)$ can be added → approval net of each reviewer's tendency

    $$
    \operatorname{logit}(p_{ij})=a_j+b_j x_{ij},\qquad
    \begin{pmatrix}a_j\\ b_j\end{pmatrix}\sim\mathcal N_2\!\left(
    \begin{pmatrix}\mu_a\\ \mu_b\end{pmatrix},
    \begin{pmatrix}\tau_a^2 & \rho\tau_a\tau_b\\ \rho\tau_a\tau_b & \tau_b^2\end{pmatrix}\right)
    $$

    - $b_j$: movie-specific effect of the viewer predictor
    - $\tau_a,\tau_b$: spread of intercepts / slopes · $\rho$: their correlation
    - Priors on $\mu_a,\mu_b,\tau_a,\tau_b,\rho$: mild, keep values plausible (half-normal on SDs, LKJ on correlation)
    """)
    return


@app.cell
def _(mo):
    vs_mua = mo.ui.slider(-1, 2, step=0.1, value=0.8, label="$\\mu_a$", show_value=True)
    vs_mub = mo.ui.slider(-1, 2, step=0.1, value=0.7, label="$\\mu_b$", show_value=True)
    vs_taua = mo.ui.slider(0.0, 1.5, step=0.05, value=0.7, label="$\\tau_a$", show_value=True)
    vs_taub = mo.ui.slider(0.0, 1.0, step=0.05, value=0.4, label="$\\tau_b$", show_value=True)
    vs_rho = mo.ui.slider(-0.9, 0.9, step=0.1, value=-0.4, label="$\\rho$", show_value=True)
    return vs_mua, vs_mub, vs_rho, vs_taua, vs_taub


@app.cell(hide_code=True)
def _(C_INK, C_MUT, C_PART, alt, mo, np, pl, vs_mua, vs_mub, vs_rho, vs_taua, vs_taub):
    _ta, _tb, _r = vs_taua.value, vs_taub.value, vs_rho.value
    _S = np.array([[_ta**2, _r * _ta * _tb], [_r * _ta * _tb, _tb**2]])
    _ab = np.random.default_rng(7).multivariate_normal([vs_mua.value, vs_mub.value], _S, size=25)
    _xs = np.linspace(-2.5, 2.5, 60)
    _curves = pl.DataFrame([(j, x, 1 / (1 + np.exp(-(a + b * x)))) for j, (a, b) in enumerate(_ab) for x in _xs],
                           schema=["movie", "x", "p"], orient="row")
    _popc = pl.DataFrame({"x": _xs, "p": 1 / (1 + np.exp(-(vs_mua.value + vs_mub.value * _xs)))})
    _c1 = (
        alt.Chart(_curves).mark_line(color=C_PART, opacity=0.35).encode(
            x=alt.X("x:Q", title="Viewer preference x (centred)"), y=alt.Y("p:Q", title="P(like)", scale=alt.Scale(domain=[0, 1])),
            detail="movie:N")
        + alt.Chart(_popc).mark_line(color=C_INK, strokeWidth=3).encode(x="x:Q", y="p:Q")
    ).properties(width=320, height=240, title="25 movies drawn from the population")
    _pts = pl.DataFrame({"a": _ab[:, 0], "b": _ab[:, 1]})
    _c2 = (
        alt.Chart(_pts).mark_point(filled=True, color=C_PART, size=50).encode(
            x=alt.X("a:Q", title="Intercept a_j", scale=alt.Scale(domain=[-2, 3.5])),
            y=alt.Y("b:Q", title="Slope b_j", scale=alt.Scale(domain=[-1.5, 3])))
        + alt.Chart(pl.DataFrame({"a": [vs_mua.value], "b": [vs_mub.value]})).mark_point(
            shape="cross", size=200, color=C_INK, filled=True).encode(x="a:Q", y="b:Q")
    ).properties(width=260, height=240, title="Their (a_j, b_j)")
    mo.vstack([
        mo.hstack([vs_mua, vs_mub, vs_rho], justify="start", gap=2),
        mo.hstack([vs_taua, vs_taub], justify="start", gap=2),
        mo.hstack([_c1, _c2], justify="start", gap=2),
        mo.md("τ = 0 → all movies share the black curve (complete pooling) · large τ → very different movies · ρ < 0 → high-baseline movies have flatter slopes"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.4 A simulated multilevel regression example

    Settings of the report (seed 2026): $J=24$ movies, 5–346 votes, $\mu_a=0.8$, $\mu_b=0.7$, $\tau_a=0.7$, $\tau_b=0.4$, $\rho=-0.4$.

    - Partial pooling: PyMC, NUTS, non-centred, priors $\mu_a\sim\mathcal N(0,1.5^2)$, $\mu_b\sim\mathcal N(0,1)$, $\tau\sim$ Half-Normal(1), LKJ(2)
    - No pooling: separate logistic regression per movie with weak $\mathcal N(0,5^2)$ priors
    - Expect: small movies move most; intercepts and slopes shrink **together** along the ellipse
    """)
    return


@app.cell
def _(minimize, np):
    # data: identical random-number sequence to multilevel_regression.py
    REG_J, REG_SEED = 24, 2026
    REG_TRUE = dict(mu_a=0.8, mu_b=0.7, tau_a=0.7, tau_b=0.4, rho=-0.4)
    _rng = np.random.default_rng(REG_SEED)
    reg_n = np.sort(np.round(np.exp(_rng.uniform(np.log(5), np.log(400), REG_J))).astype(int))
    _t = REG_TRUE
    _S = np.array([[_t["tau_a"] ** 2, _t["rho"] * _t["tau_a"] * _t["tau_b"]],
                   [_t["rho"] * _t["tau_a"] * _t["tau_b"], _t["tau_b"] ** 2]])
    _ab = _rng.multivariate_normal([_t["mu_a"], _t["mu_b"]], _S, size=REG_J)
    reg_a_true, reg_b_true = _ab[:, 0], _ab[:, 1]
    reg_movie = np.repeat(np.arange(REG_J), reg_n)
    reg_x = _rng.normal(0, 1, reg_n.sum())
    reg_z = _rng.binomial(1, 1 / (1 + np.exp(-(reg_a_true[reg_movie] + reg_b_true[reg_movie] * reg_x))))

    def _sep_fit(xj, zj):
        def nlp(p):
            e = p[0] + p[1] * xj
            return np.sum(np.logaddexp(0, e) - zj * e) + np.sum(p**2) / (2 * 5.0**2)
        return minimize(nlp, np.zeros(2), method="BFGS").x

    reg_sep = np.array([_sep_fit(reg_x[reg_movie == j], reg_z[reg_movie == j]) for j in range(REG_J)])
    return REG_J, REG_SEED, REG_TRUE, reg_a_true, reg_b_true, reg_movie, reg_n, reg_sep, reg_x, reg_z


@app.cell
def _(mo):
    reg_mode = mo.ui.dropdown(
        {"Quick (4 chains × 500 draws)": 500, "Report (4 chains × 2000 draws)": 2000},
        value="Quick (4 chains × 500 draws)", label="Sampling",
    )
    reg_run = mo.ui.run_button(label="Fit hierarchical model with PyMC")
    mo.hstack([reg_mode, reg_run], justify="start")
    return reg_mode, reg_run


@app.cell
def _(REG_J, REG_SEED, mo, np, reg_mode, reg_movie, reg_run, reg_x, reg_z):
    mo.stop(not reg_run.value, mo.callout(mo.md("Press **Fit hierarchical model** to run NUTS (quick ≈ 30 s, report ≈ 2 min)."), kind="info"))
    import pymc as pm

    with mo.status.spinner(title="Sampling with NUTS…"):
        with pm.Model():
            _mu_a = pm.Normal("mu_a", 0, 1.5)
            _mu_b = pm.Normal("mu_b", 0, 1.0)
            _chol, _corr, _sds = pm.LKJCholeskyCov("chol", n=2, eta=2.0, sd_dist=pm.HalfNormal.dist(1.0, shape=2),
                                                   compute_corr=True)
            _q = pm.Normal("q", 0, 1, shape=(REG_J, 2))
            _eff = pm.Deterministic("eff", pm.math.stack([_mu_a, _mu_b]) + pm.math.dot(_q, _chol.T))
            pm.Deterministic("tau_a", _sds[0]); pm.Deterministic("tau_b", _sds[1]); pm.Deterministic("rho", _corr[0, 1])
            pm.Bernoulli("z", logit_p=_eff[reg_movie, 0] + _eff[reg_movie, 1] * reg_x, observed=reg_z)
            _idata = pm.sample(reg_mode.value, tune=reg_mode.value, chains=4, cores=1, target_accept=0.95,
                               random_seed=REG_SEED, progressbar=False)

    _post = _idata.posterior
    reg_draws = {k: _post[k].values for k in ["mu_a", "mu_b", "tau_a", "tau_b", "rho"]}   # (chain, draw)
    reg_eff = _post["eff"].values                                                         # (chain, draw, J, 2)
    reg_div = int(np.asarray(_idata.sample_stats["diverging"].values).sum())
    return reg_div, reg_draws, reg_eff


@app.cell
def _(np):
    def split_rhat(x):
        """Basic split-R-hat for an array of shape (chain, draw)."""
        c, d = x.shape
        h = d // 2
        s = np.concatenate([x[:, :h], x[:, h:2 * h]], axis=0)
        w = s.var(axis=1, ddof=1).mean()
        b = h * s.mean(axis=1).var(ddof=1)
        return float(np.sqrt(((h - 1) / h * w + b / h) / w))
    return (split_rhat,)


@app.cell(hide_code=True)
def _(REG_TRUE, mo, np, pl, reg_a_true, reg_b_true, reg_div, reg_draws, reg_eff, reg_sep, split_rhat):
    reg_a_hat = reg_eff[..., 0].reshape(-1, reg_eff.shape[2]).mean(0)
    reg_b_hat = reg_eff[..., 1].reshape(-1, reg_eff.shape[2]).mean(0)
    _rows = []
    for _k, _lab in [("mu_a", "μ_a mean intercept"), ("mu_b", "μ_b mean slope"), ("tau_a", "τ_a SD of intercepts"),
                     ("tau_b", "τ_b SD of slopes"), ("rho", "ρ correlation")]:
        _d = reg_draws[_k]
        _f = _d.ravel()
        _rows.append((_lab, REG_TRUE[_k], round(float(_f.mean()), 2), round(float(_f.std()), 2),
                      f"[{np.quantile(_f, 0.025):.2f}, {np.quantile(_f, 0.975):.2f}]", round(split_rhat(_d), 3)))
    _tab = pl.DataFrame(_rows, schema=["Parameter", "True", "Post. mean", "Post. SD", "95% interval", "R-hat"], orient="row")
    _rh = max(split_rhat(reg_eff[..., j, k]) for j in range(reg_eff.shape[2]) for k in range(2))
    _mse = lambda e, t: float(np.mean((e - t) ** 2))
    _m = pl.DataFrame({
        "Effect": ["Intercepts a_j", "Slopes b_j"],
        "No pooling": [round(_mse(reg_sep[:, 0], reg_a_true), 3), round(_mse(reg_sep[:, 1], reg_b_true), 3)],
        "Complete pooling": [round(_mse(reg_draws["mu_a"].mean(), reg_a_true), 3), round(_mse(reg_draws["mu_b"].mean(), reg_b_true), 3)],
        "Partial pooling": [round(_mse(reg_a_hat, reg_a_true), 3), round(_mse(reg_b_hat, reg_b_true), 3)],
    })
    mo.vstack([
        mo.md("Report mode reproduces **Table 2** of the report; quick mode gives nearly the same numbers."),
        mo.md(f"**Diagnostics:** divergences = **{reg_div}**, largest split R-hat over all movie effects = **{_rh:.3f}**"),
        md_table(_tab),
        mo.md("**Mean squared error against the true movie effects**"),
        md_table(_m),
        mo.md("- Intercepts: partial pooling ≈ halves the error\n- Slopes: no gain here — Movie 13's unusually steep slope (2.31) is pulled to ≈1.4 → shrinkage helps **on average**, can overshrink unusual movies"),
    ])
    return reg_a_hat, reg_b_hat


@app.cell
def _(REG_J, mo, reg_n):
    reg_pick = mo.ui.multiselect(
        options={f"Movie {j+1} (n={reg_n[j]})": j for j in range(REG_J)},
        value=["Movie 1 (n=5)", "Movie 7 (n=22)", "Movie 24 (n=346)"], label="Movies in panel (b)", max_selections=4,
    )
    return (reg_pick,)


@app.cell(hide_code=True)
def _(
    C_EXTRA, C_INK, C_MUT, C_NO, C_PART, alt, mo, np, pl, reg_a_hat, reg_b_hat, reg_draws, reg_eff, reg_n, reg_pick,
    reg_sep,
):
    _ma, _mb = float(reg_draws["mu_a"].mean()), float(reg_draws["mu_b"].mean())
    _ta, _tb, _r = (float(reg_draws[k].mean()) for k in ["tau_a", "tau_b", "rho"])
    _S = np.array([[_ta**2, _r * _ta * _tb], [_r * _ta * _tb, _tb**2]])
    _L = np.linalg.cholesky(_S)
    _t = np.linspace(0, 2 * np.pi, 120)
    _ell = []
    for _lev in (0.5, 0.9):
        _c = np.sqrt(-2 * np.log(1 - _lev))
        _pts = np.array([_ma, _mb])[:, None] + _c * _L @ np.vstack([np.cos(_t), np.sin(_t)])
        _ell += [(f"{int(_lev*100)}%", i, x, y) for i, (x, y) in enumerate(_pts.T)]
    _ell = pl.DataFrame(_ell, schema=["level", "order", "a", "b"], orient="row")
    _df = pl.DataFrame({"movie": np.arange(1, len(reg_n) + 1), "n": reg_n, "a_sep": reg_sep[:, 0], "b_sep": reg_sep[:, 1],
                        "a_hat": reg_a_hat, "b_hat": reg_b_hat})
    _pa = (
        alt.Chart(_ell).mark_line(color=C_MUT, strokeWidth=1).encode(
            x="a:Q", y="b:Q", order="order:Q", strokeDash=alt.StrokeDash("level:N", title="Population", legend=alt.Legend(orient="bottom")))
        + alt.Chart(_df).mark_rule(color=C_MUT, opacity=0.6).encode(x="a_sep:Q", y="b_sep:Q", x2="a_hat:Q", y2="b_hat:Q")
        + alt.Chart(_df).mark_point(filled=True, color=C_NO, opacity=0.9).encode(
            x=alt.X("a_sep:Q", title="Intercept a_j"), y=alt.Y("b_sep:Q", title="Slope b_j"),
            size=alt.Size("n:Q", scale=alt.Scale(type="log", range=[20, 160]), legend=None),
            tooltip=["movie", "n", alt.Tooltip("a_sep:Q", format=".2f"), alt.Tooltip("b_sep:Q", format=".2f")])
        + alt.Chart(_df).mark_point(filled=True, color=C_PART, opacity=0.95).encode(
            x="a_hat:Q", y="b_hat:Q", size=alt.Size("n:Q", scale=alt.Scale(type="log", range=[20, 160]), legend=None),
            tooltip=["movie", "n", alt.Tooltip("a_hat:Q", format=".2f"), alt.Tooltip("b_hat:Q", format=".2f")])
        + alt.Chart(pl.DataFrame({"a": [_ma], "b": [_mb]})).mark_point(shape="cross", size=200, filled=True, color=C_INK).encode(x="a:Q", y="b:Q")
    ).properties(width=330, height=290, title="(a) Orange: separate fits → blue: partial pooling")

    _xs = np.linspace(-2.5, 2.5, 80)
    _flat = reg_eff.reshape(-1, reg_eff.shape[2], 2)[::10]
    _rows = [("Population", "Population", x, float(1 / (1 + np.exp(-(_ma + _mb * x))))) for x in _xs]
    for _j in reg_pick.value:
        _lab = f"Movie {_j+1} (n={reg_n[_j]})"
        _pd = 1 / (1 + np.exp(-(_flat[:, _j, 0][:, None] + _flat[:, _j, 1][:, None] * _xs[None, :])))
        _rows += [(_lab, "Partial pooling", x, float(p)) for x, p in zip(_xs, _pd.mean(0))]
        _rows += [(_lab, "No pooling", x, float(1 / (1 + np.exp(-(reg_sep[_j, 0] + reg_sep[_j, 1] * x))))) for x in _xs]
    _cd = pl.DataFrame(_rows, schema=["movie", "method", "x", "p"], orient="row")
    _pb = alt.Chart(_cd).mark_line(strokeWidth=2).encode(
        x=alt.X("x:Q", title="Viewer preference x"), y=alt.Y("p:Q", title="P(like)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("movie:N", scale=alt.Scale(domain=["Population"] + [f"Movie {j+1} (n={reg_n[j]})" for j in reg_pick.value],
                                                   range=[C_INK] + C_EXTRA), title=None, legend=alt.Legend(orient="bottom", columns=2)),
        strokeDash=alt.StrokeDash("method:N", scale=alt.Scale(domain=["Partial pooling", "No pooling", "Population"],
                                                               range=[[1, 0], [5, 3], [1, 0]]), title=None),
        detail="method:N",
    ).properties(width=300, height=290, title="(b) Predicted approval (averaged over draws)")
    mo.vstack([reg_pick, alt.hconcat(_pa, _pb).resolve_scale(color="independent", strokeDash="independent"),
               mo.md("- Hover points for movie details · pick movies for panel (b)\n- Small movies: dashed (separate) curve is extreme, solid (hierarchical) curve is pulled toward the population\n- Large movies: both curves coincide")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 7. Posterior Computation and Assessment

    - Beta–binomial with fixed $\alpha,\beta$: closed form
    - Gamma hyperpriors: no closed form → grid over $(\alpha,\beta)$ or MCMC
    - Hierarchical logistic regression: Hamiltonian Monte Carlo / NUTS

    $$
    q_j\sim\mathcal N(0,1),\qquad a_j=\mu_a+\tau_a q_j
    $$

    - Non-centred form: sample the standardised $q_j$, then scale by $\tau_a$ and shift by $\mu_a$ → easier geometry when groups carry little information
    - Check: several chains, $\hat R\approx1$, effective sample size, **no divergences**
    - Convergence ≠ correct model → posterior predictive checks, leave-one-out cross-validation
    - Leave out single votes → known movies · leave out whole movies → new titles
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 8. Limitations and Discussion

    - Like/dislike discards rating intensity → ordinal hierarchical model possible
    - Votes not truly independent: self-selection, advertising, campaigns
    - $\theta_j$ describes the **voting population**, not objective film quality
    - One population distribution can **overshrink** genuinely unusual movies (Movie 13)
    - $\operatorname{Gamma}(2,1)$ pulls $\kappa$ down (≈5 vs true 10) → sensitivity analysis needed
    - Hierarchy = relation of parameters · partial pooling = information sharing · shrinkage = resulting adjustment

    ---
    ## 9. Conclusion

    - Each estimate = compromise between the movie's own data and the population
    - Few votes → strong shrinkage · many votes → almost none
    - Hyperpriors let the data choose how strong the shrinkage is
    - Same idea extends to regression: intercepts and slopes shrink jointly
    - Shrinkage helps **on average**, but can pull unusual groups too far
    - Careful priors, reliable computation and model checking remain essential
    """)
    return


if __name__ == "__main__":
    app.run()
