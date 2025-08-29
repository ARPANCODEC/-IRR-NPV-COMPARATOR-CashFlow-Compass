import math
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

# =============================
# App Setup
# =============================
st.set_page_config(page_title="IRR & NPV Analyzer", page_icon="💹", layout="wide")
st.title("💹 IRR & NPV Analyzer (Newton’s Method + NPV)")

# =============================
# Finance Utilities
# =============================
def npv(rate: float, cashflows: List[float]) -> float:
    """
    Net Present Value of a discrete cashflow stream.
    rate: discount rate as decimal (e.g., 0.05 for 5%)
    cashflows: list where index = year, cashflows[0] is Year 0
    """
    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))


def _d_npv(rate: float, cashflows: List[float]) -> float:
    """
    Derivative of NPV with respect to the rate (for Newton's method).
    """
    return sum(-t * cf / ((1 + rate) ** (t + 1)) for t, cf in enumerate(cashflows))


def irr_newton(
    cashflows: List[float],
    guess: float = 0.1,
    tol: float = 1e-10,
    max_iter: int = 1000,
    fallback_guesses: Tuple[float, ...] = (-0.5, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0),
    log_steps: bool = True,
) -> Tuple[Optional[float], Optional[pd.DataFrame]]:
    """
    IRR via Newton-Raphson. Returns (irr, steps_df).
    steps_df columns: ['iter', 'r', 'npv(r)', "npv'(r)", 'r_next', 'error']
    Returns (None, steps_df or None) if it fails to converge.

    Notes:
    - Assumes conventional project CF pattern: one sign change (outflow then inflows).
    - For tricky CFs, tries multiple starting guesses.
    """
    cols = ['iter', 'r', 'npv(r)', "npv'(r)", 'r_next', 'error']

    def _solve(start):
        rows: List[Dict[str, Any]] = []
        r = start
        for k in range(1, max_iter + 1):
            f = npv(r, cashflows)
            df = _d_npv(r, cashflows)
            if df == 0 or not math.isfinite(df):
                if log_steps:
                    rows.append({'iter': k, 'r': r, 'npv(r)': f, "npv'(r)": df, 'r_next': np.nan, 'error': np.nan})
                return None, pd.DataFrame(rows, columns=cols) if log_steps else None
            r_next = r - f / df
            err = abs(r_next - r)
            if log_steps:
                rows.append({'iter': k, 'r': r, 'npv(r)': f, "npv'(r)": df, 'r_next': r_next, 'error': err})
            if not math.isfinite(r_next):
                return None, pd.DataFrame(rows, columns=cols) if log_steps else None
            if err < tol:
                return r_next, pd.DataFrame(rows, columns=cols) if log_steps else None
            r = r_next
        return None, pd.DataFrame(rows, columns=cols) if log_steps else None

    # First try user guess, then fallbacks
    attempts = (guess,) + tuple(g for g in fallback_guesses if g != guess)
    for g in attempts:
        ans, steps = _solve(g)
        if ans is not None:
            return ans, steps
    return None, steps  # return the last steps if available


def parse_cf(s: str) -> List[float]:
    """Parse comma/newline separated numbers into a list of floats."""
    vals = []
    for part in s.replace("\n", ",").split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    return vals


def profitability_index(rate: float, cashflows: List[float]) -> Optional[float]:
    """
    PI = PV(inflows) / |initial outflow|
    """
    if not cashflows:
        return None
    c0 = cashflows[0]
    if c0 >= 0:
        return None
    pv_total = npv(rate, cashflows)
    pv_inflows = pv_total - c0  # because c0 already included in NPV
    if -c0 == 0:
        return None
    return pv_inflows / (-c0)


def discounted_table(cashflows: List[float], rate: float) -> pd.DataFrame:
    """
    Build a DCF table with columns:
    Year, Cash Flow, Discount Factor, Present Value, Cumulative PV
    """
    rows = []
    cumulative = 0.0
    for t, cf in enumerate(cashflows):
        df = 1.0 / ((1 + rate) ** t) if math.isfinite(rate) else (1.0 if t == 0 else 0.0)
        pv = cf * df
        cumulative += pv
        rows.append({
            "Year": t,
            "Cash Flow": cf,
            "Discount Factor": df,
            "Present Value": pv,
            "Cumulative PV": cumulative
        })
    return pd.DataFrame(rows)


# =============================
# Sidebar Controls
# =============================
st.sidebar.header("Global Settings")
rate = st.sidebar.number_input(
    "Discount rate for NPV (%)",
    min_value=-50.0,
    max_value=200.0,
    value=5.0,
    step=0.25,
) / 100.0
user_guess = st.sidebar.number_input(
    "Initial IRR guess (%)",
    min_value=-90.0,
    max_value=500.0,
    value=10.0,
    step=1.0,
) / 100.0
precision = st.sidebar.number_input("Decimal places", min_value=2, max_value=8, value=4, step=1)

with st.expander("What’s with the negative signs?"):
    st.markdown(
        """
- By convention, **cash outflows** (your investment) are **negative** and **cash inflows** are **positive**.
- Example: `[-100, 30, 30, 30, 30, 30]` means invest 100 now (Year 0), then receive 30 each year after.
- **IRR** is the discount rate that makes **NPV = 0**.  
- **NPV** at your chosen rate tells you the present value created today.
        """
    )

# =============================
# Defaults (PJ2 pre-filled)
# =============================
default_p1 = "-100, 30, 30, 30, 30, 30"
default_p2 = "-150, 42, 42, 42, 42, 42"

tab1, tab2, tab3 = st.tabs(["Project 1", "Project 2", "Compare"])

# =============================
# Project 1
# =============================
with tab1:
    st.subheader("Project 1")
    p1_text = st.text_area("Cash flows (comma- or newline-separated)", value=default_p1, height=100)
    show_p1_calcs = st.checkbox("Show detailed calculations (DCF table & Newton steps)", value=True, key="p1_show")
    if p1_text.strip():
        try:
            p1 = parse_cf(p1_text)
            irr1, steps1 = irr_newton(p1, guess=user_guess)
            npv1 = npv(rate, p1)
            pi1 = profitability_index(rate, p1)
            c1, c2, c3 = st.columns(3)
            c1.metric("NPV", f"{npv1:.{precision}f}")
            c2.metric("IRR", f"{(irr1*100):.{precision}f}%" if irr1 is not None else "No convergence")
            c3.metric("Profitability Index", f"{pi1:.{precision}f}" if pi1 is not None else "n/a")
            st.caption(f"Cash flows parsed: {p1}")

            if show_p1_calcs:
                st.markdown("#### Discounted Cash Flow (DCF) Table")
                df1 = discounted_table(p1, rate)
                st.dataframe(
                    df1.style.format({
                        "Cash Flow": f"{{:.{precision}f}}",
                        "Discount Factor": f"{{:.{precision}f}}",
                        "Present Value": f"{{:.{precision}f}}",
                        "Cumulative PV": f"{{:.{precision}f}}",
                    }),
                    use_container_width=True
                )
                st.markdown("#### Newton’s Method Iterations for IRR")
                if steps1 is not None and not steps1.empty:
                    st.dataframe(
                        steps1.style.format({
                            'r': f"{{:.{precision}f}}",
                            'npv(r)': f"{{:.{precision}f}}",
                            "npv'(r)": f"{{:.{precision}f}}",
                            'r_next': f"{{:.{precision}f}}",
                            'error': f"{{:.{precision}e}}",
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("No iteration log available (did not converge or no steps logged).")

        except Exception as e:
            st.error(f"Error parsing/processing Project 1: {e}")

# =============================
# Project 2
# =============================
with tab2:
    st.subheader("Project 2")
    p2_text = st.text_area("Cash flows (comma- or newline-separated)", value=default_p2, height=100)
    show_p2_calcs = st.checkbox("Show detailed calculations (DCF table & Newton steps)", value=True, key="p2_show")
    if p2_text.strip():
        try:
            p2 = parse_cf(p2_text)
            irr2, steps2 = irr_newton(p2, guess=user_guess)
            npv2 = npv(rate, p2)
            pi2 = profitability_index(rate, p2)
            c1, c2, c3 = st.columns(3)
            c1.metric("NPV", f"{npv2:.{precision}f}")
            c2.metric("IRR", f"{(irr2*100):.{precision}f}%" if irr2 is not None else "No convergence")
            c3.metric("Profitability Index", f"{pi2:.{precision}f}" if pi2 is not None else "n/a")
            st.caption(f"Cash flows parsed: {p2}")

            if show_p2_calcs:
                st.markdown("#### Discounted Cash Flow (DCF) Table")
                df2 = discounted_table(p2, rate)
                st.dataframe(
                    df2.style.format({
                        "Cash Flow": f"{{:.{precision}f}}",
                        "Discount Factor": f"{{:.{precision}f}}",
                        "Present Value": f"{{:.{precision}f}}",
                        "Cumulative PV": f"{{:.{precision}f}}",
                    }),
                    use_container_width=True
                )
                st.markdown("#### Newton’s Method Iterations for IRR")
                if steps2 is not None and not steps2.empty:
                    st.dataframe(
                        steps2.style.format({
                            'r': f"{{:.{precision}f}}",
                            'npv(r)': f"{{:.{precision}f}}",
                            "npv'(r)": f"{{:.{precision}f}}",
                            'r_next': f"{{:.{precision}f}}",
                            'error': f"{{:.{precision}e}}",
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("No iteration log available (did not converge or no steps logged).")

        except Exception as e:
            st.error(f"Error parsing/processing Project 2: {e}")

# =============================
# Compare
# =============================
with tab3:
    st.subheader("Side-by-side Comparison")
    try:
        p1 = parse_cf(p1_text)
        p2 = parse_cf(p2_text)

        irr1, _ = irr_newton(p1, guess=user_guess, log_steps=False)
        irr2, _ = irr_newton(p2, guess=user_guess, log_steps=False)
        npv1 = npv(rate, p1)
        npv2 = npv(rate, p2)
        pi1 = profitability_index(rate, p1)
        pi2 = profitability_index(rate, p2)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Project 1")
            st.write(f"Cash flows: `{p1}`")
            st.write(f"**NPV @ {rate*100:.2f}%:** {npv1:.{precision}f}")
            st.write("**IRR:** {}".format(f"{irr1*100:.{precision}f}%" if irr1 is not None else "No convergence"))
            st.write("**Profitability Index:** {}".format(f"{pi1:.{precision}f}" if pi1 is not None else "n/a"))

        with col2:
            st.markdown("### Project 2")
            st.write(f"Cash flows: `{p2}`")
            st.write(f"**NPV @ {rate*100:.2f}%:** {npv2:.{precision}f}")
            st.write("**IRR:** {}".format(f"{irr2*100:.{precision}f}%" if irr2 is not None else "No convergence"))
            st.write("**Profitability Index:** {}".format(f"{pi2:.{precision}f}" if pi2 is not None else "n/a"))

        st.divider()
        st.markdown("### Decision Hints")
        st.markdown(
            f"""
- At a cost of capital of **{rate*100:.2f}%**, choose the **higher NPV** if projects are **mutually exclusive**.
- If **capital is constrained**, prefer the project with a **higher Profitability Index** (and often higher IRR).
- A project is acceptable if **NPV > 0** at your discount rate.
            """
        )

    except Exception as e:
        st.error(f"Comparison error: {e}")

# =============================
# Footer (branded)
# =============================
st.markdown("---")
# Sticky footer styling
st.markdown(
    """
    <style>
      .custom-footer {
        position: fixed;
        left: 0; right: 0; bottom: 0;
        background: white;
        border-top: 1px solid #e6e6e6;
        padding: 10px 16px;
        text-align: center;
        font-size: 14px;
        color: grey;
        z-index: 9999;
      }
      /* Give bottom padding so content isn't hidden behind sticky footer */
      .block-container { padding-bottom: 70px; }
      /* Make dataframes use full width */
      .stDataFrame { width: 100% !important; }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div class="custom-footer">
        Made by <b>Arpan Ari (arpancodec)</b> &nbsp; | &nbsp; All Rights Reserved © 2025
    </div>
    """,
    unsafe_allow_html=True
)
