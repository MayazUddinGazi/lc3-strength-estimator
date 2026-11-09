"""LC3 Strength Estimator: Streamlit app. Run locally with `streamlit run app.py`."""
from pathlib import Path
import io
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from lc3_model import (LC3Model, kappa_from_kaolinite, domain_checks, predict_table, error_summary,
                       BATCH_COLUMNS, BATCH_HELP)

st.set_page_config(page_title='LC3 Strength Estimator', layout='wide')
HERE = Path(__file__).parent


# ---------------- access gate (set APP_PASSWORD in Streamlit secrets; no secret = open)
def gate():
    import os
    pw = os.environ.get('APP_PASSWORD')
    if not pw and ((HERE / '.streamlit' / 'secrets.toml').exists() or (Path.home() / '.streamlit' / 'secrets.toml').exists()):
        try:
            pw = st.secrets.get('APP_PASSWORD')
        except Exception:
            pw = None
    if not pw:
        return
    if st.session_state.get('auth_ok'):
        return
    st.title('LC3 Strength Estimator')
    st.caption('Research prototype (University of Miami). Access is restricted while the work is under review.')
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

st.title('LC3 Strength Estimator')
st.caption('Hierarchical Bayesian Bolomey model fitted to 437 mix–age observations from ten verified studies '
           '(six research groups). Research prototype for screening and validation, not for design.')

tab1, tab2, tab3 = st.tabs(['Single mix', 'Batch validation (CSV)', 'Model and reliability'])

# ====================================================================== TAB 1: single mix
with tab1:
    with st.sidebar:
        st.header('Binder (mass %)')
        cem = st.number_input('Portland cement (clinker based)', 0.0, 100.0, 50.0, 0.5)
        cc = st.number_input('Calcined clay', 0.0, 100.0, 30.0, 0.5)
        ls = st.number_input('Limestone', 0.0, 100.0, 15.0, 0.5)
        gy = st.number_input('Gypsum', 0.0, 20.0, 5.0, 0.5)
        st.header('Mix and curing')
        wb = st.number_input('Water-to-binder ratio', 0.10, 1.00, 0.50, 0.01)
        age = st.number_input('Age (days)', 0.5, 730.0, 28.0, 1.0)
        temp = st.number_input('Curing temperature (°C)', 0.0, 80.0, 20.0, 1.0)
        st.header('Calcined clay quality')
        basis = st.selectbox('Kaolinite reported as', ['Kaolinite of raw clay', 'Metakaolin of calcined clay', 'Not known'])
        kv = None if basis == 'Not known' else st.number_input('Content (%)', 0.0, 100.0, 50.0, 1.0)
        st.header('Local reference (optional)')
        use_c = st.checkbox('I have a Portland cement control from the same laboratory', value=True)
        cage = st.radio('Control tested at', ['28 d', '7 d'], horizontal=True, disabled=not use_c,
                        help='A 28-d control recovered the full gain in the validation; a 7-d control about 80% of it.')
        cfc = st.number_input(f'Measured {cage} strength of the control (MPa)', 1.0, 200.0, 50.0, 0.1, disabled=not use_c)
        cwb = st.number_input('Water-to-binder ratio of the control', 0.10, 1.00, wb, 0.01, disabled=not use_c)

    total = cem + cc + ls + gy
    if total <= 0:
        st.error('Enter the binder composition.'); st.stop()
    kappa = None
    if cc > 0 and kv is not None:
        kappa = kappa_from_kaolinite(kv, 'calcined' if basis.startswith('Metakaolin') else 'raw')
    mix = dict(c=(cem + gy) / total, m=(cc + ls) / total, w=wb, rho=(cc + ls) / total, kappa=kappa)
    x = dict(cem=cem, cc=cc, ls=ls, gy=gy, sum=total, wb=wb, age=age, temp=temp)
    c_age_d = 28 if cage.startswith('28') else 7
    control = dict(fc=cfc, w=cwb, age=c_age_d) if use_c else None

    checks = domain_checks(x, mix)
    bad = [m for lev, m in checks if lev == 'bad']

    if bad:
        st.error('No prediction: ' + ' '.join(bad))
    else:
        rn = model.predict(mix, age, temp)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader('New laboratory, no reference')
            st.metric('80% interval', f"{rn['lo']:.1f} – {rn['hi']:.1f} MPa")
            st.write(f"Median {rn['median']:.1f} MPa (interval ±{(rn['hi']-rn['lo'])/2/rn['median']*100:.0f}%)")
        with c2:
            st.subheader('Anchored on your control')
            if control:
                rc = model.predict(mix, age, temp, control)
                exp_c = model.control_expected(cwb, temp, c_age_d)
                st.metric('80% interval', f"{rc['lo']:.1f} – {rc['hi']:.1f} MPa")
                st.write(f"Median {rc['median']:.1f} MPa ({rc['median']-rn['median']:+.1f} MPa vs no reference; "
                         f"interval ±{(rc['hi']-rc['lo'])/2/rc['median']*100:.0f}%)")
                st.caption(f'The model expects {exp_c:.1f} MPa for your {c_age_d}-d control; you measured {cfc:.1f} MPa '
                           f'({(cfc/exp_c-1)*100:+.0f}% for this laboratory).')
            else:
                st.info('Tick the local reference and enter your control strength to anchor the prediction.')
        st.write(f"Replacement ρ = {mix['rho']:.2f} · κ = {'unknown' if kappa is None else f'{kappa:.2f}'} · "
                 f"equivalent age {rn['te']:.1f} d · efficiency factor k = {rn['k']:.2f}")

        ages = np.exp(np.linspace(0, np.log(365), 30))
        rows = []
        for a in ages:
            r = model.predict(mix, a, temp); rows.append(dict(age=a, case='New laboratory', **r))
            if control:
                r = model.predict(mix, a, temp, control); rows.append(dict(age=a, case='Anchored on control', **r))
        df = pd.DataFrame(rows)
        xs = alt.X('age:Q', scale=alt.Scale(type='log', domain=[1, 365], nice=False), title='Age (days, log scale)')
        col = alt.Color('case:N', scale=alt.Scale(domain=['New laboratory', 'Anchored on control'], range=['#7a8589', '#0d6b72']), title=None)
        band = alt.Chart(df).mark_area(opacity=0.25).encode(xs, alt.Y('lo:Q', title='Compressive strength (MPa)'), alt.Y2('hi:Q'), col)
        line = alt.Chart(df).mark_line(strokeWidth=2.5).encode(xs, alt.Y('median:Q'), col)
        rule = alt.Chart(pd.DataFrame({'age': [age]})).mark_rule(color='black').encode(x='age:Q')
        st.subheader('Strength development')
        st.altair_chart((band + line + rule).properties(height=340), use_container_width=True)

    st.subheader('Inside the data the model learned from?')
    for lev, msg in checks:
        (st.success if lev == 'ok' else st.warning if lev == 'warn' else st.error)(msg)

# ====================================================================== TAB 2: batch validation
with tab2:
    st.markdown('Upload a CSV with one row per mix and age. Columns (order does not matter; blanks allowed where noted):')
    st.dataframe(pd.DataFrame({'column': BATCH_COLUMNS, 'meaning': [BATCH_HELP[c] for c in BATCH_COLUMNS]}),
                 hide_index=True, use_container_width=True)
    tmpl = pd.DataFrame([dict(mix_id='LC3-50 example', cement_pct=50, calcined_clay_pct=30, limestone_pct=15, gypsum_pct=5,
                              w_b=0.50, age_d=28, cure_T_C=20, kaolinite_pct=50, kaolinite_basis='raw', fc_measured_MPa=45.0,
                              control_fc_MPa=50.0, control_age_d=28, control_w_b=0.50, group='my lab')], columns=BATCH_COLUMNS)
    cA, cB = st.columns(2)
    cA.download_button('Download empty template (CSV)', tmpl.to_csv(index=False).encode(), 'lc3_batch_template.csv', 'text/csv')
    ex_path = HERE / 'example_validation.csv'
    if ex_path.exists():
        cB.download_button('Download example: the 437 training observations with their 28-d controls',
                           ex_path.read_bytes(), 'example_training_data.csv', 'text/csv')
    ext_path = HERE / 'example_external_test.csv'
    if ext_path.exists():
        st.download_button('Download example: external test set (12 rows from two studies not in the training data; '
                           'Canbek et al. 2022, Hossain et al. 2026)', ext_path.read_bytes(), 'example_external_test.csv', 'text/csv')
    up = st.file_uploader('CSV file', type=['csv'])
    use_example = st.checkbox('Use the training-data example instead (in-sample check: the model has seen these data)')
    use_ext = st.checkbox('Use the external test set instead (out-of-sample: the model has not seen these data)')
    src = None
    if up is not None:
        src = pd.read_csv(up)
    elif use_ext and ext_path.exists():
        src = pd.read_csv(ext_path)
        st.info('Canbek et al. 2022: mortar cubes, w/b 0.40, metakaolin ≥95% kaolinite; values at 3 and 7 d and for two mixes at 28 d are derived from the '
                'percentage changes reported in the paper (±1–2 MPa); no Portland cement control was reported. Hossain et al. 2026: one optimized LC3 concrete, '
                '100 mm cubes, 28 d, kaolinite not reported, no control. Both studies were overpredicted without a reference; see the flags column.')
    elif use_example and ex_path.exists():
        src = pd.read_csv(ex_path)
    if src is not None:
        missing = [c for c in ['cement_pct', 'calcined_clay_pct', 'limestone_pct', 'w_b', 'age_d'] if c not in src.columns]
        if missing:
            st.error('Missing required columns: ' + ', '.join(missing))
        else:
            for c in BATCH_COLUMNS:
                if c not in src.columns: src[c] = np.nan
            with st.spinner(f'Predicting {len(src)} rows…'):
                res = predict_table(model, src)
            st.success(f'{len(res)} rows predicted.')
            show = ['mix_id', 'group', 'age_d', 'fc_measured_MPa', 'pred_new_lab', 'lo_new_lab', 'hi_new_lab',
                    'pred_anchored', 'lo_anchored', 'hi_anchored', 'rho', 'kappa', 'k', 'flags']
            st.dataframe(res[show].round(2), hide_index=True, use_container_width=True, height=320)
            st.download_button('Download results (CSV)', res.to_csv(index=False).encode(), 'lc3_predictions.csv', 'text/csv')
            if res['fc_measured_MPa'].notna().any():
                st.subheader('Agreement with your measurements')
                by = 'group' if res['group'].notna().any() else None
                summ = error_summary(res, by)
                st.dataframe(summ.round(2), hide_index=True, use_container_width=True)
                d = res[res['fc_measured_MPa'].notna()]
                long = []
                for tag, p in [('New laboratory', 'pred_new_lab'), ('Anchored on control', 'pred_anchored')]:
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
                    alt.Color('case:N', scale=alt.Scale(domain=['New laboratory', 'Anchored on control'], range=['#7a8589', '#0d6b72']), title=None),
                    tooltip=['mix', 'age', 'measured', 'predicted', 'case'])
                st.altair_chart((diag + pts).properties(height=420, width=420), use_container_width=False)
                st.caption('MAE in the published validation (source withheld): 6.6 MPa without a reference, 4.7 MPa with a 28-d control, '
                           '5.1 MPa with a 7-d control. In-sample agreement with the example file is better than this and is not a test.')

# ====================================================================== TAB 3: notes
with tab3:
    st.markdown("""
**Model.** f = A₂₈ β(t) [(c + k m)/w − B], with k = sp(α₀ + α₁κ + α₂ρ)·[1 − exp(−(t/τ)^γ)] and a source effect *u* and a mix effect *v*
on the log scale. Without a reference, *u* is drawn from its fitted spread (σᵤ = 0.15); with a Portland cement control from the
same laboratory, *u* is estimated from the control and removed through a multiplicative offset. The control must share cement,
water-to-binder ratio, specimen, curing and testing with the LC3 mixes; otherwise it corrects only the part of the level they share.

**Data.** 437 mix–age observations, ten studies, six research groups; w/b 0.30–0.55, replacement up to 0.80, 1–365 d, no fly ash or slag,
no heat curing. Mostly mortar; two concrete studies.

**Accuracy on withheld studies (leave-one-source-out).** MAE 6.6 MPa (about 16%) with no reference; 4.7 MPa with a 28-d control
(improved 8 of 10 studies); 5.1 MPa with a 7-d control (81% of the 28-d gain); no gain from a control at 3 d or earlier.
Coverage of the 80% interval: 81% with one study withheld, 73% with a whole research group withheld, 54% on three later-added studies.
Early ages (1–3 d) transfer worst.

**Use.** Screening of candidate mixes and comparison with your own measurements (Batch validation tab). Not a design tool;
predictions are not a substitute for testing.

Model and validation: M. U. Gazi and X. Shi, University of Miami. Manuscript under review (2026).
""")
