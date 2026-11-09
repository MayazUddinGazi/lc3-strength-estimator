"""LC3 strength model for the web app: the hierarchical Bayesian Bolomey model fitted to all primary data.

Same equations as run_bayes.py predict():
    k  = sp20(a0 + a1*kappa + a2*rho) * (1 - exp(-(te/tau)**gamma))
    X  = (c + k*m)/w - B
    mu = lnA28 + s*(1 - sqrt(28/te)) + ln(sp50(X) + 1e-6)
    ln fc = mu + u_source + v_mix (+ noise)
The posterior draws come from m2_post_primary.npz (4000 draws, NUTS, PyMC).
"""
import numpy as np

R_GAS, T_REF, EA, SD_LOG = 8.314, 293.15, 40000.0, 0.055


def sp(x, k):
    """Smooth positive function ln(1 + e^(kx))/k, as in the model."""
    return np.logaddexp(0, k * x) / k


def equivalent_age(age_d, temp_c, ea=EA):
    """Arrhenius equivalent age at 20 degC."""
    return age_d * np.exp(-ea / R_GAS * (1 / (temp_c + 273.15) - 1 / T_REF))


def kappa_from_kaolinite(pct, basis):
    """Metakaolin mass fraction of the calcined clay from a raw-clay kaolinite content (or a calcined-basis value)."""
    k = pct / 100.0
    return k if basis == 'calcined' else 0.8604 * k / (1 - 0.1396 * k)


class LC3Model:
    def __init__(self, npz_path, seed=2026):
        z = np.load(npz_path)
        self.p = {k: z[k] for k in z.files}
        n = len(self.p['lnA28']); rng = np.random.default_rng(seed)
        # fixed random numbers, so the same input always gives the same answer
        self.z_u, self.z_v, self.z_vc, self.z_e, self.z_ec = rng.normal(size=(5, n))
        self.kappa_prior = rng.beta(4, 3, n)          # clay quality when kaolinite is unknown
        self.n = n

    def mu(self, c, m, w, rho, kappa, te):
        p = self.p
        k = sp(p['a0'] + p['a1'] * kappa + p['a2'] * rho, 20) * (1 - np.exp(-(te / p['tau']) ** p['gamma']))
        X = (c + k * m) / w - p['B']
        return p['lnA28'] + p['s'] * (1 - np.sqrt(28 / te)) + np.log(sp(X, 50) + 1e-6), k

    def predict(self, mix, age, temp_c, control=None):
        """mix: dict c, m, w, rho, kappa (None = unknown).
        control: None or dict fc (MPa), w, age (d; 28 by default, 7 allowed) of a Portland cement control
        from the same laboratory. Returns median, 10% and 90% strength (MPa), median k and the equivalent age."""
        p = self.p
        te = equivalent_age(age, temp_c)
        if mix['m'] <= 0:
            kap = 0.0
        else:
            kap = self.kappa_prior if mix['kappa'] is None else mix['kappa']
        mu, k = self.mu(mix['c'], mix['m'], mix['w'], mix['rho'], kap, te)
        noise = np.sqrt(p['sigma_e'] ** 2 + SD_LOG ** 2)
        if control is None:
            # new laboratory: draw a new source effect u and a new mix effect v
            base = mu + p['sigma_u'] * self.z_u + p['sigma_v'] * self.z_v
            full = base + noise * self.z_e
        else:
            # anchored on the measured control: the source effect cancels, mix effects and noise of both remain
            c_age = control.get('age', 28)
            mu_c, _ = self.mu(1.0, 0.0, control['w'], 0.0, 0.0, equivalent_age(c_age, temp_c))
            base = np.log(control['fc']) + (mu - mu_c) + p['sigma_v'] * (self.z_v - self.z_vc)
            full = base + noise * (self.z_e - self.z_ec)
        return dict(median=float(np.median(np.exp(base))), lo=float(np.percentile(np.exp(full), 10)),
                    hi=float(np.percentile(np.exp(full), 90)), k=float(np.median(k)), te=float(te))

    def control_expected(self, w, temp_c, age=28):
        """What the model expects for a Portland cement control at the given age in a new laboratory (median, MPa)."""
        return self.predict(dict(c=1.0, m=0.0, w=w, rho=0.0, kappa=0.0), age, temp_c)['median']


def domain_checks(x, mix):
    """Compare inputs with the fitted data. Returns a list of (level, message); level is ok, warn or bad."""
    out = []
    wb, rho = x['wb'], mix['rho']
    if wb < 0.25 or wb > 0.70: out.append(('bad', f'w/b {wb:.2f} is far outside the data (0.30–0.55).'))
    elif wb < 0.30 or wb > 0.55: out.append(('warn', f'w/b {wb:.2f} is outside the fitted range 0.30–0.55.'))
    else: out.append(('ok', f'w/b {wb:.2f} within 0.30–0.55.'))
    if rho > 0.85: out.append(('bad', f'Replacement {rho*100:.0f}% is beyond the data (maximum 80%).'))
    elif rho > 0.80: out.append(('warn', f'Replacement {rho*100:.0f}% is slightly above 80%.'))
    else: out.append(('ok', f'Replacement {rho*100:.0f}% within 0–80%.'))
    age = x['age']
    if age < 0.5 or age > 730: out.append(('bad', f'Age {age:g} d is outside the range the model can extrapolate to.'))
    elif age < 1 or age > 365: out.append(('warn', f'Age {age:g} d is outside 1–365 d.'))
    else: out.append(('ok', f'Age {age:g} d within 1–365 d.'))
    T = x['temp']
    if T < 5 or T > 60: out.append(('bad', f'Curing at {T:g} °C is outside the scope (no heat curing in the data).'))
    elif T < 18 or T > 32: out.append(('warn', f'Curing at {T:g} °C: the data were cured at 20–30 °C.'))
    else: out.append(('ok', f'Curing at {T:g} °C within 20–30 °C.'))
    if x['cc'] > 0:
        if mix['kappa'] is None: out.append(('warn', 'Kaolinite unknown: clay quality drawn from Beta(4,3), which widens the interval.'))
        elif mix['kappa'] < 0.15: out.append(('warn', f'κ = {mix["kappa"]:.2f} is below the lowest clay in the data (0.15).'))
        else: out.append(('ok', f'κ = {mix["kappa"]:.2f} within 0.15–1.00.'))
    if x['gy'] / x['sum'] > 0.08: out.append(('warn', 'Gypsum above 8% of binder; the data contain up to about 5%.'))
    if x['cc'] > 0 and mix['kappa'] is not None and mix['kappa'] > 0.8 and wb < 0.45:
        out.append(('warn', f'High-purity metakaolin (κ = {mix["kappa"]:.2f}) at w/b {wb:.2f}: the kaolinite effect was learned mainly from one '
                            'research group at w/b 0.50, and an external test on such a system (metakaolin ≥95%, w/b 0.40) was overpredicted by about 45%. '
                            'Treat the prediction as an upper bound and anchor it on a control if possible.'))
    if x['cc'] > 0 and x['cc'] + x['ls'] > 0:
        r = x['cc'] / (x['cc'] + x['ls'])
        if r < 0.55 or r > 0.78:
            out.append(('warn', f'Calcined clay share of the SCM fraction is {r:.2f}; the model assigns one efficiency to clay + limestone and was fitted to a near-constant 2:1 ratio (0.67), so it predicts the same strength for different clay:limestone splits.'))
    return out


# ---------------------------------------------------------------- batch helpers
BATCH_COLUMNS = ['mix_id', 'cement_pct', 'calcined_clay_pct', 'limestone_pct', 'gypsum_pct', 'w_b', 'age_d',
                 'cure_T_C', 'kaolinite_pct', 'kaolinite_basis', 'fc_measured_MPa',
                 'control_fc_MPa', 'control_age_d', 'control_w_b', 'group']
BATCH_HELP = {
    'mix_id': 'any label', 'cement_pct': 'Portland cement, mass % of binder', 'calcined_clay_pct': 'mass % of binder',
    'limestone_pct': 'mass % of binder', 'gypsum_pct': 'added gypsum, mass % of binder (0 if none)',
    'w_b': 'water-to-binder ratio', 'age_d': 'test age, days', 'cure_T_C': 'curing temperature, °C (20 if blank)',
    'kaolinite_pct': 'kaolinite of the raw clay or metakaolin of the calcined clay, % (blank = unknown)',
    'kaolinite_basis': 'raw or calcined (raw if blank)', 'fc_measured_MPa': 'measured strength, optional',
    'control_fc_MPa': 'measured Portland cement control of the same laboratory, optional',
    'control_age_d': '28 or 7 (28 if blank)', 'control_w_b': 'w/b of the control (same as the mix if blank)',
    'group': 'laboratory / study / batch label for the summary, optional'}


def predict_table(model, df):
    """Row-wise prediction for a batch table in the BATCH_COLUMNS format. Returns a copy with result columns."""
    import pandas as pd
    out = df.copy()
    cols = {c: [] for c in ['pred_new_lab', 'lo_new_lab', 'hi_new_lab', 'pred_anchored', 'lo_anchored', 'hi_anchored',
                            'kappa', 'rho', 'k', 'flags']}
    for _, r in df.iterrows():
        cem, cc, ls = float(r['cement_pct']), float(r['calcined_clay_pct']), float(r['limestone_pct'])
        gy = float(r['gypsum_pct']) if pd.notna(r.get('gypsum_pct')) else 0.0
        tot = cem + cc + ls + gy
        wb, age = float(r['w_b']), float(r['age_d'])
        T = float(r['cure_T_C']) if pd.notna(r.get('cure_T_C')) else 20.0
        kap = None
        if cc > 0 and pd.notna(r.get('kaolinite_pct')):
            basis = str(r.get('kaolinite_basis', 'raw')).strip().lower()
            kap = kappa_from_kaolinite(float(r['kaolinite_pct']), 'calcined' if basis.startswith('calc') else 'raw')
        mix = dict(c=(cem + gy) / tot, m=(cc + ls) / tot, w=wb, rho=(cc + ls) / tot, kappa=kap)
        x = dict(cem=cem, cc=cc, ls=ls, gy=gy, sum=tot, wb=wb, age=age, temp=T)
        checks = domain_checks(x, mix)
        flags = '; '.join(m for lev, m in checks if lev != 'ok')
        rn = model.predict(mix, age, T)
        cols['pred_new_lab'].append(rn['median']); cols['lo_new_lab'].append(rn['lo']); cols['hi_new_lab'].append(rn['hi'])
        if pd.notna(r.get('control_fc_MPa')):
            cage = float(r['control_age_d']) if pd.notna(r.get('control_age_d')) else 28.0
            cwb = float(r['control_w_b']) if pd.notna(r.get('control_w_b')) else wb
            rc = model.predict(mix, age, T, dict(fc=float(r['control_fc_MPa']), w=cwb, age=cage))
            cols['pred_anchored'].append(rc['median']); cols['lo_anchored'].append(rc['lo']); cols['hi_anchored'].append(rc['hi'])
        else:
            for c in ('pred_anchored', 'lo_anchored', 'hi_anchored'): cols[c].append(np.nan)
        cols['kappa'].append(np.nan if kap is None else kap); cols['rho'].append(mix['rho']); cols['k'].append(rn['k']); cols['flags'].append(flags)
    for c, v in cols.items(): out[c] = v
    return out


def error_summary(res, by=None):
    """MAE, MAPE, bias and 80%-interval coverage against fc_measured_MPa, overall or per group."""
    import pandas as pd
    d = res[pd.notna(res['fc_measured_MPa'])].copy()
    if d.empty: return None
    def stats(g):
        o = {}
        y = g['fc_measured_MPa'].astype(float)
        for tag, p, lo, hi in [('new lab', 'pred_new_lab', 'lo_new_lab', 'hi_new_lab'),
                               ('anchored', 'pred_anchored', 'lo_anchored', 'hi_anchored')]:
            m = pd.notna(g[p])
            if m.sum() == 0: continue
            e = g.loc[m, p] - y[m]
            o[f'n ({tag})'] = int(m.sum())
            o[f'MAE MPa ({tag})'] = float(np.mean(np.abs(e)))
            o[f'MAPE % ({tag})'] = float(np.mean(np.abs(e) / y[m]) * 100)
            o[f'bias MPa ({tag})'] = float(np.mean(e))
            o[f'80% coverage % ({tag})'] = float(np.mean((y[m] >= g.loc[m, lo]) & (y[m] <= g.loc[m, hi])) * 100)
        return pd.Series(o)
    if by and by in d.columns and d[by].notna().any():
        parts = []
        for g, sub in d.groupby(by):
            row = stats(sub); row[by] = g; parts.append(row)
        return pd.DataFrame(parts)[[by] + [c for c in parts[0].index if c != by]]
    return stats(d).to_frame('all rows').T
