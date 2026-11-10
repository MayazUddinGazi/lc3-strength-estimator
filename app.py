"""LC3 Strength Estimator: Streamlit app. Run locally with `streamlit run app.py`."""
from pathlib import Path
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from lc3_model import (LC3Model, kappa_from_kaolinite, domain_checks, predict_table, error_summary,
                       BATCH_COLUMNS, BATCH_HELP)

st.set_page_config(page_title='LC3 Strength Estimator', layout='wide', initial_sidebar_state='expanded')
HERE = Path(__file__).parent

# ---------------- light styling
st.markdown("""
<style>
h1 {font-size: 2.0rem !important; margin-bottom: 0.2rem;}
h2 {font-size: 1.35rem !important; margin-top: 0.6rem;}
h3 {font-size: 1.1rem !important;}
[data-testid="stMetricValue"] {font-size: 1.9rem;}
.step {border: 1px solid #d9dee1; border-radius: 10px; padding: 12px 14px; background: #f7f9f9; height: 100%;}
.step b {color: #0d6b72;}
.note {font-size: 0.9rem; color: #4a5559;}
.legend span {display: inline-block; padding: 2px 10px; border-radius: 6px; margin-right: 8px; font-size: 0.85rem;}
.lg-ok {background: #dff3e4;} .lg-warn {background: #fff3cd;} .lg-bad {background: #f8d7da;}
section[data-testid="stSidebar"] .stCaption {margin-top: -6px;}
</style>
""", unsafe_allow_html=True)


# ---------------- access gate (set APP_PASSWORD in Streamlit secrets; no secret = open)
def gate():
    import os
    pw = os.environ.get('APP_PASSWORD')
    if not pw and ((HERE / '.streamlit' / 'secrets.toml').exists() or (Path.home() / '.streamlit' / 'secrets.toml').exists()):
        try:
            pw = st.secrets.get('APP_PASSWORD')
        except Exception:
            pw = None
    if not pw or st.session_state.get('auth_ok'):
        return
    st.title('LC3 Strength Estimator')
    given = st.text_input('Access code', type='password')
    if given and given == pw:
        st.session_state['auth_ok'] = True
        st.rerun()
    elif given:
        st.error('Wrong access code.')
    st.stop()


gate()


@st.cache_resource
def load_model():
    return LC3Model(HERE / 'm2_post_primary.npz')


model = load_model()

# ====================================================================== header
st.title('LC3 Strength Estimator')
st.markdown('Estimates the **compressive strength** of a limestone calcined clay cement (LC3) mix at any age, '
            'with an uncertainty range, from its composition. Research prototype (University of Miami); '
            'for screening and checking, not for design.')

with st.expander('How to use this tool (read once)', expanded=True):
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="step"><b>1 · Describe your mix</b><br>In the left panel enter the binder (cement, calcined clay, '
                'limestone, gypsum in % by mass), the water-to-binder ratio, the age and the curing temperature, and the '
                'kaolinite content of your clay if you know it.</div>', unsafe_allow_html=True)
    c2.markdown('<div class="step"><b>2 · Add your control (optional, recommended)</b><br>If your laboratory has tested a '
                'plain Portland cement mortar or concrete at 28 d (or 7 d), enter its strength. The model uses it to '
                'calibrate the prediction to your cement, specimens and testing.</div>', unsafe_allow_html=True)
    c3.markdown('<div class="step"><b>3 · Read the result</b><br>You get a strength <i>range</i> (80% interval), not one '
                'number. Check the colored list below the chart: <span style="background:#dff3e4">green</span> = your '
                'input is inside the data the model learned from, <span style="background:#fff3cd">yellow</span> = at the '
                'edge, read with care, <span style="background:#f8d7da">red</span> = too far, no prediction.</div>',
                unsafe_allow_html=True)
    st.markdown('<p class="note">Have your own test results? Use the <b>Compare with my data</b> tab: upload a table of '
                'mixes with measured strengths and see how close the model gets, mix by mix.</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(['Predict one mix', 'Compare with my data (CSV)', 'About the model'])

# ====================================================================== sidebar inputs
with st.sidebar:
    st.header('Your mix')
    st.caption('Binder composition, % by mass of binder. The four values are normalized, so they need not sum to 100.')
    cem = st.number_input('Portland cement', 0.0, 100.0, 50.0, 0.5, help='Clinker-based cement, including the gypsum already in it. Typical LC3-50: 50–55%.')
    cc = st.number_input('Calcined clay', 0.0, 100.0, 30.0, 0.5, help='Metakaolin or calcined kaolinitic clay. Typical LC3-50: 30%.')
    ls = st.number_input('Limestone', 0.0, 100.0, 15.0, 0.5, help='Limestone powder (filler). Typical LC3-50: 15%. The data were fitted mostly to a 2:1 clay:limestone ratio.')
    gy = st.number_input('Added gypsum', 0.0, 20.0, 5.0, 0.5, help='Gypsum added on top of the cement, % of binder. Enter 0 if none was added.')
    st.caption('Mix and curing')
    wb = st.number_input('Water-to-binder ratio (w/b)', 0.10, 1.00, 0.50, 0.01, help='Total water divided by total binder (cement + clay + limestone + gypsum). Data cover 0.30–0.55.')
    age = st.number_input('Test age (days)', 0.5, 730.0, 28.0, 1.0, help='Age at which you want the strength. Data cover 1–365 d; 28 d is the most reliable.')
    temp = st.number_input('Curing temperature (°C)', 0.0, 80.0, 20.0, 1.0, help='Water or moist curing temperature. Data cover 20–30 °C; no heat curing.')
    st.caption('Clay quality')
    basis = st.selectbox('Kaolinite reported as', ['Kaolinite of raw clay', 'Metakaolin of calcined clay', 'Not known'],
                         help='Kaolinite % of the raw clay (from XRD or TGA before calcination), or metakaolin % of the calcined clay. If unknown, the model assumes a typical clay and the range gets wider.')
    kv = None if basis == 'Not known' else st.number_input('Content (%)', 0.0, 100.0, 50.0, 1.0,
                                                           help='Typical LC3 clays: 40–60% kaolinite in the raw clay. Commercial metakaolin: 90–95%.')
    st.header('Your control (optional)')
    st.caption('A plain Portland cement mix made and tested in the same laboratory as the LC3 mix.')
    use_c = st.checkbox('I have a Portland cement control', value=True,
                        help='Same cement, similar w/b, same specimen size, curing and testing as your LC3 mix. The model learns your laboratory level from it.')
    cage = st.radio('Control tested at', ['28 d', '7 d'], horizontal=True, disabled=not use_c,
                    help='A 28-d control gives the full correction; a 7-d control gives about 80% of it, three weeks earlier.')
    cfc = st.number_input(f'Measured {cage} strength of the control (MPa)', 1.0, 200.0, 50.0, 0.1, disabled=not use_c)
    cwb = st.number_input('w/b of the control', 0.10, 1.00, wb, 0.01, disabled=not use_c, help='Usually the same as the LC3 mix.')

total = cem + cc + ls + gy
kappa = None
if cc > 0 and kv is not None:
    kappa = kappa_from_kaolinite(kv, 'calcined' if basis.startswith('Metakaolin') else 'raw')
mix = dict(c=(cem + gy) / total, m=(cc + ls) / total, w=wb, rho=(cc + ls) / total, kappa=kappa) if total > 0 else None
x = dict(cem=cem, cc=cc, ls=ls, gy=gy, sum=total, wb=wb, age=age, temp=temp)
c_age_d = 28 if cage.startswith('28') else 7
control = dict(fc=cfc, w=cwb, age=c_age_d) if use_c else None

# ====================================================================== TAB 1: single mix
with tab1:
    if total <= 0:
        st.error('Enter the binder composition in the left panel.'); st.stop()
    checks = domain_checks(x, mix)
    bad = [m for lev, m in checks if lev == 'bad']
    warn = [m for lev, m in checks if lev == 'warn']

    if bad:
        st.error('**No prediction.** Your mix is too far from the data the model learned from: ' + ' '.join(bad))
    else:
        rn = model.predict(mix, age, temp)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader('Without your control')
            st.caption('What the model expects for this mix in an unknown laboratory.')
            st.metric(f'Strength at {age:g} d, 80% range', f"{rn['lo']:.0f} – {rn['hi']:.0f} MPa")
            st.markdown(f"Most likely about **{rn['median']:.1f} MPa**. The range is wide (±{(rn['hi']-rn['lo'])/2/rn['median']*100:.0f}%) "
                        f"because cement, specimens and testing differ between laboratories.")
        with c2:
            st.subheader('Calibrated with your control')
            if control:
                rc = model.predict(mix, age, temp, control)
                exp_c = model.control_expected(cwb, temp, c_age_d)
                off = (cfc / exp_c - 1) * 100
                st.caption('The same prediction, shifted to the level of your laboratory.')
                st.metric(f'Strength at {age:g} d, 80% range', f"{rc['lo']:.0f} – {rc['hi']:.0f} MPa",
                          f"{rc['median']-rn['median']:+.1f} MPa vs without control", delta_color='off')
                st.markdown(f"Most likely about **{rc['median']:.1f} MPa**. Your control gave {cfc:.1f} MPa where the model expected "
                            f"{exp_c:.1f} MPa, so your laboratory runs **{off:+.0f}%** against the published data; the LC3 prediction is shifted by the same factor.")
                if abs(off) > 25:
                    st.warning('Your control differs from the model expectation by more than 25%. Check that the control shares the cement, w/b, specimen size and curing of the LC3 mix; otherwise the correction may be misleading.')
            else:
                st.info('Tick **I have a Portland cement control** in the left panel and enter its strength. '
                        'In the validation, one 28-d control reduced the error from 6.8 to 4.7 MPa.')
        if warn:
            st.warning('Read this prediction with care: ' + ' '.join(warn))

        st.markdown(f'<p class="note">Model quantities: replacement ρ = {mix["rho"]:.2f} · κ (metakaolin fraction of the clay) = '
                    f'{"assumed, unknown" if kappa is None else f"{kappa:.2f}"} · equivalent age {rn["te"]:.1f} d · efficiency factor of the clay–limestone fraction k = {rn["k"]:.2f}</p>',
                    unsafe_allow_html=True)

        ages = np.exp(np.linspace(0, np.log(365), 30))
        rows = []
        for a in ages:
            r = model.predict(mix, a, temp); rows.append(dict(age=a, case='Without control', **r))
            if control:
                r = model.predict(mix, a, temp, control); rows.append(dict(age=a, case='Calibrated with control', **r))
        df = pd.DataFrame(rows)
        xs = alt.X('age:Q', scale=alt.Scale(type='log', domain=[1, 365], nice=False), title='Age (days, log scale)')
        col = alt.Color('case:N', scale=alt.Scale(domain=['Without control', 'Calibrated with control'], range=['#7a8589', '#0d6b72']), title=None)
        band = alt.Chart(df).mark_area(opacity=0.25).encode(xs, alt.Y('lo:Q', title='Compressive strength (MPa)'), alt.Y2('hi:Q'), col)
        line = alt.Chart(df).mark_line(strokeWidth=2.5).encode(xs, alt.Y('median:Q'), col)
        rule = alt.Chart(pd.DataFrame({'age': [age]})).mark_rule(color='black').encode(x='age:Q')
        st.subheader('Strength development')
        st.caption('Line: most likely value. Band: 80% range. Vertical line: the age you asked for. Values before 3 d are the least reliable.')
        st.altair_chart((band + line + rule).properties(height=320), use_container_width=True)

    st.subheader('Is your mix inside the data the model learned from?')
    st.markdown('<div class="legend"><span class="lg-ok">green: inside the data</span><span class="lg-warn">yellow: at the edge, read with care</span>'
                '<span class="lg-bad">red: too far, no prediction</span></div>', unsafe_allow_html=True)
    for lev, msg in checks:
        (st.success if lev == 'ok' else st.warning if lev == 'warn' else st.error)(msg)

# ====================================================================== TAB 2: batch validation
with tab2:
    st.markdown('Upload your own test results and see how close the model gets, mix by mix. Three steps:')
    s1, s2, s3 = st.columns(3)
    s1.markdown('<div class="step"><b>1 · Get the template</b><br>Download the empty CSV below. One row = one mix tested at one age.</div>', unsafe_allow_html=True)
    s2.markdown('<div class="step"><b>2 · Fill it in</b><br>Required: cement, calcined clay, limestone (%), w/b, age. Add the measured strength to compare, '
                'your control strength to calibrate, and a <i>group</i> label (laboratory or study) to get a summary per group. Leave what you do not know blank.</div>', unsafe_allow_html=True)
    s3.markdown('<div class="step"><b>3 · Upload and read</b><br>You get the prediction for every row, a summary per group (mean error, bias, how many results fell inside the 80% range) and a measured-vs-predicted plot.</div>', unsafe_allow_html=True)

    with st.expander('Column meanings'):
        st.dataframe(pd.DataFrame({'column': BATCH_COLUMNS, 'meaning': [BATCH_HELP[c] for c in BATCH_COLUMNS]}),
                     hide_index=True, use_container_width=True)
    tmpl = pd.DataFrame([dict(mix_id='LC3-50 example', cement_pct=50, calcined_clay_pct=30, limestone_pct=15, gypsum_pct=5,
                              w_b=0.50, age_d=28, cure_T_C=20, kaolinite_pct=50, kaolinite_basis='raw', fc_measured_MPa=45.0,
                              control_fc_MPa=50.0, control_age_d=28, control_w_b=0.50, group='my lab')], columns=BATCH_COLUMNS)
    cA, cB, cC = st.columns(3)
    cA.download_button('Download empty template (CSV)', tmpl.to_csv(index=False).encode(), 'lc3_batch_template.csv', 'text/csv', use_container_width=True)
    ex_path = HERE / 'example_validation.csv'; ext_path = HERE / 'example_external_test.csv'
    if ex_path.exists():
        cB.download_button('Example: training data (437 rows)', ex_path.read_bytes(), 'example_training_data.csv', 'text/csv', use_container_width=True)
    if ext_path.exists():
        cC.download_button('Example: two unseen studies (12 rows)', ext_path.read_bytes(), 'example_external_test.csv', 'text/csv', use_container_width=True)

    up = st.file_uploader('Upload your CSV', type=['csv'])
    st.caption('No file yet? Try one of the examples:')
    e1, e2 = st.columns(2)
    use_example = e1.checkbox('Training data (the model has seen these; shows the in-sample fit)')
    use_ext = e2.checkbox('Two unseen studies (Canbek et al. 2022, Hossain et al. 2026; shows an out-of-sample test)')
    src = None
    if up is not None:
        src = pd.read_csv(up)
    elif use_ext and ext_path.exists():
        src = pd.read_csv(ext_path)
        st.info('Canbek et al. 2022: mortar cubes, w/b 0.40, metakaolin ≥95% kaolinite; values at 3 and 7 d and for two mixes at 28 d are derived from the '
                'percentage changes reported in the paper (±1–2 MPa); no Portland cement control was reported. Hossain et al. 2026: one optimized LC3 concrete, '
                '100 mm cubes, 28 d, kaolinite not reported, no control. Both are overpredicted without a reference; the flags column says why.')
    elif use_example and ex_path.exists():
        src = pd.read_csv(ex_path)
    if src is not None:
        missing = [c for c in ['cement_pct', 'calcined_clay_pct', 'limestone_pct', 'w_b', 'age_d'] if c not in src.columns]
        if missing:
            st.error('Missing required columns: ' + ', '.join(missing) + '. Use the template above.')
        else:
            for c in BATCH_COLUMNS:
                if c not in src.columns: src[c] = np.nan
            with st.spinner(f'Predicting {len(src)} rows…'):
                res = predict_table(model, src)
            st.success(f'{len(res)} rows predicted.')
            st.subheader('Predictions, row by row')
            st.caption('pred_new_lab / lo / hi: prediction and 80% range without a control. pred_anchored: with your control (blank if none). '
                       'flags: anything outside the data the model learned from.')
            show = ['mix_id', 'group', 'age_d', 'fc_measured_MPa', 'pred_new_lab', 'lo_new_lab', 'hi_new_lab',
                    'pred_anchored', 'lo_anchored', 'hi_anchored', 'rho', 'kappa', 'k', 'flags']
            st.dataframe(res[show].round(2), hide_index=True, use_container_width=True, height=320)
            st.download_button('Download results (CSV)', res.to_csv(index=False).encode(), 'lc3_predictions.csv', 'text/csv')
            if res['fc_measured_MPa'].notna().any():
                st.subheader('How close is the model to your measurements?')
                st.markdown('<p class="note"><b>MAE</b>: average error in MPa (lower is better). <b>MAPE</b>: the same in %. '
                            '<b>bias</b>: positive = the model predicts too high on average, negative = too low. '
                            '<b>80% coverage</b>: share of your results that fell inside the predicted range; about 80% means the ranges are honest, '
                            'much less means the model is too confident for your data. Compare the <i>new lab</i> and <i>anchored</i> columns to see what your control buys.</p>',
                            unsafe_allow_html=True)
                by = 'group' if res['group'].notna().any() else None
                summ = error_summary(res, by)
                st.dataframe(summ.round(2), hide_index=True, use_container_width=True)
                d = res[res['fc_measured_MPa'].notna()]
                long = []
                for tag, p in [('Without control', 'pred_new_lab'), ('Calibrated with control', 'pred_anchored')]:
                    m = d[p].notna()
                    if m.any():
                        long.append(pd.DataFrame({'measured': d.loc[m, 'fc_measured_MPa'], 'predicted': d.loc[m, p],
                                                  'case': tag, 'mix': d.loc[m, 'mix_id'].astype(str), 'age': d.loc[m, 'age_d']}))
                long = pd.concat(long)
                mx = float(max(long['measured'].max(), long['predicted'].max()) * 1.05)
                diag = alt.Chart(pd.DataFrame({'x': [0, mx], 'y': [0, mx]})).mark_line(color='black', strokeDash=[4, 4]).encode(x='x:Q', y='y:Q')
                pts = alt.Chart(long).mark_circle(size=60, opacity=0.7).encode(
                    alt.X('measured:Q', title='Measured strength (MPa)', scale=alt.Scale(domain=[0, mx])),
                    alt.Y('predicted:Q', title='Predicted strength (MPa)', scale=alt.Scale(domain=[0, mx])),
                    alt.Color('case:N', scale=alt.Scale(domain=['Without control', 'Calibrated with control'], range=['#7a8589', '#0d6b72']), title=None),
                    tooltip=['mix', 'age', 'measured', 'predicted', 'case'])
                st.caption('Points on the dashed line are perfect predictions; above the line the model predicts too high, below it too low. Hover a point for the mix name.')
                st.altair_chart((diag + pts).properties(height=420, width=420), use_container_width=False)
                st.caption('For reference, in the published validation (whole studies withheld) the error was 6.6 MPa without a control, '
                           '4.7 MPa with a 28-d control and 5.1 MPa with a 7-d control.')

# ====================================================================== TAB 3: notes
with tab3:
    st.markdown("""
**What the model is.** A physics-based strength law (Bolomey) with an efficiency factor for the calcined clay–limestone fraction that
depends on age, clay quality (κ) and replacement level, fitted as a hierarchical Bayesian model to 437 verified mix–age observations
from ten published studies (six research groups). Each study has its own *level* (cement, specimens, curing, testing); the model
separates this level from the effect of composition.

**Why a control helps.** The level of an unknown laboratory is the largest source of error. A plain Portland cement mix tested in the
same laboratory measures that level, and the model shifts its LC3 prediction by the same factor. The control must share cement,
w/b, specimen size, curing and testing with the LC3 mix; otherwise it corrects only part of the difference.

**Data the model learned from.** w/b 0.30–0.55, replacement up to 80%, kaolinite κ 0.15–1.0 (mostly 0.4–0.6), 1–365 d, curing 20–30 °C,
no fly ash or slag, no heat curing; mostly mortar, two concrete studies; clay:limestone close to 2:1.

**How accurate it is.** On studies the model had not seen: average error 6.6 MPa (about 16%) without a control, 4.7 MPa with a 28-d
control (better in 8 of 10 studies), 5.1 MPa with a 7-d control. The 80% range contained 81% of results when one study was withheld,
but only 54–73% for truly new research groups, so treat the range as a minimum. Early ages (1–3 d) are predicted worst. For
high-purity metakaolin (κ > 0.8) at w/b below 0.45 the model overpredicted an external study by about 45%; the tool warns in this case.

**What it is not.** Not a design tool and not a substitute for testing. Use it to screen candidate mixes, to see what a control
would tell you, and to compare with your own results.

Model and validation: M. U. Gazi and X. Shi, University of Miami. Manuscript under review (2026). Source code: github.com/MayazUddinGazi/lc3-strength-estimator.
""")
