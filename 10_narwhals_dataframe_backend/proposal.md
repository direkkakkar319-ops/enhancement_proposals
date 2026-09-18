## Replace Hardcoded `pandas` Assumptions with `narwhals` for DataFrame-Agnostic Data Ingestion

Contributors: **[`@direkkakkar319`](https://github.com/direkkakkar319-ops)**


### Introduction

This proposal addresses the narwhals part of **[`Issue 3395`](https://github.com/pgmpy/pgmpy/issues/3395)**.
It complements the maintainer's architectural design in **[`PR #14`](https://github.com/pgmpy/enhancement_proposals/pull/14)** by **[`@ankurankan`](https://github.com/ankurankan)**

pgmpy currently hardcodes `pandas.DataFrame` as the only accepted tabular input type across its entire public API-parameter estimators, conditional independence tests, structure scores, causal discovery algorithms, and model-level methods like `fit()`, `predict()`, and `simulate()`.

This creates several practical problems like:
1. **Ecosystem lock-in.** Users working with **[`Polars`​](https://github.com/pola-rs/polars)** , **[`PyArrow`​](https://github.com/apache/arrow)** , **[`cuDF`​](https://github.com/rapidsai/cudf)** or **[`Modin`​](https://github.com/modin-project/modin)** must manually convert their data to pandas before calling any pgmpy function, and convert back afterward.

2. **Unnecessary data copies.** These conversions often force a full materialization of the data in memory. For GPU-resident data (e.g., `cuDF` DataFrames), this means an expensive device-to-host transfer and memory duplication. For lazy frames (`Polars Lazy`, `Modin`), it forces eager evaluation.

3. **Maintenance burden.** The codebase uses pandas-specific idioms (`pd.api.types.is_integer_dtype`, `pd.factorize`, `groupby(...).size().unstack()`, `pd.MultiIndex`, `pd.Categorical`, etc.) in many source files. Every new feature must be written against `Pandas` internals, and any upstream  `Pandas` deprecation requires a sweep across the entire codebase.

4. **Growing ecosystem fragmentation.** `Polars`, adoption is accelerating rapidly. Libraries that accept only `Pandas` are increasingly perceived as legacy. Supporting multiple DataFrame backends positions `Pgmpy` for long-term relevance.


### Proposed Solution

Introduce `narwhals` as a lightweight compatibility layer between pgmpy's internal logic and the user's DataFrame library.
Narwhals provides a Polars-inspired API that wraps native DataFrames *without copying data*. 
It supports pandas, Polars, PyArrow, cuDF, Modin, and other backends, and has zero required dependencies — it only uses libraries the user already has installed.

**Core pattern:** At each public API entry point (estimator `.fit()`, CI test `__init__`, model `.fit()` / `.predict()` / `.simulate()`), wrap the incoming DataFrame using `nw.from_native(df)`, perform all internal operations using the narwhals API, and convert back to the user's native format via `.to_native()` before returning.

**What does NOT change:**
- The internal factor/inference layer continues to operate on numpy arrays (or torch tensors via `array-api-compat` in the future). The narwhals boundary stops at the point where data is converted into `numpy` arrays for numerical computation.
- All existing pandas-based user code continues to work identically — pandas is a first-class narwhals backend.

**Dependency:** `narwhals` is a zero-dependency, lightweight package. We will have to add it as a core dependency in `pyproject.toml`.

### Alternative Solutions

#### 1. Convert Everything to `pandas` at the Boundary

Accept any DataFrame type, immediately convert to pandas via `nw.from_native(DataFrame).to_pandas()`, then proceed with existing pandas code unchanged.

- **Pros:** Minimal code changes. Existing internal logic stays untouched.
- **Cons:** Defeats the purpose. Users still pay the conversion cost (memory copy, GPU-to-CPU transfer). Polars lazy evaluation is forced eager. No real multi-backend support — just syntactic sugar over `.to_pandas()`.

#### 2. Use `narwhals` for Transparent, Zero-Copy Wrapping **(Proposed Solution)**

Wrap DataFrames at entry points, operate through narwhals' common API internally, return native types to the user.

- **Pros:** Zero-copy. True multi-backend support. Single code path to maintain. Narwhals is well-maintained and adopted by scikit-lego, Hamilton, and other ecosystem libraries. Lightweight dependency.
- **Cons:** Requires migrating internal pandas idioms to narwhals equivalents. Some advanced pandas operations (e.g., `pd.MultiIndex`, `pd.Categorical`) have no direct narwhals equivalent and require restructuring. One-time migration effort.

**Conclusion:** Option 3 is the clear winner. It provides genuine multi-backend support with minimal ongoing maintenance.

### Details of Proposed Solution

#### Migration Strategy

Implementation proceeds **bottom-up**: migrate the pandas-heaviest utility layer first (since everything depends on it), then work upward through the estimator base, concrete estimators, CI tests, and finally the model-level APIs. This will ensure each layer can be tested in isolation before its consumers are migrated.

```
Implementation order (bottom-up)          Runtime data flow (top-down)
============================              ==========================

Step 1: Utility Layer                     User passes DataFrame
  tabular.py, utils.py                            |
      ^                                           v
      |                               --------------------------
Step 2: Estimator Base Layer           | Layer 5: Model APIs      |  model.fit(), predict()
  base.py (_initialize_fit)            --------------------------
      ^                                           |
      |                               --------------------------
Step 3: Concrete Estimator             | Layer 4: Causal Discovery|  PC, GES, HillClimb
  discrete_mle.py (pilot)              --------------------------
      ^                                           |
      |                               --------------------------
Step 4: CI Tests                       | Layer 3: CI Tests        |  ChiSquare, FisherZ
  power_divergence.py                  --------------------------
      ^                                           |
      |                               --------------------------
Step 5: Model APIs                     | Layer 2: Estimator Base  |  nw.from_native(df) here
  DiscreteBayesianNetwork.py           --------------------------
                                                  |
                                       --------------------------
                                       | Layer 1: Utilities       |  narwhals DataFrame ops
                                       |  (preprocess_data,       |
                                       |   get_state_counts, etc.)|
                                       --------------------------
                                                  |
                                       --------------------------
                                       | Numpy Boundary           |  .to_numpy() -- stops here
                                       |  (TabularCPD, bincount)  |
```

#### Key Files and Transformations

The files below are listed in **implementation order** - each depends only on layers already migrated.

---

**Step 1a: **[`tabular.py`](pgmpy/utils/tabular.py)** - **[`collect_state_names()`](pgmpy/utils/tabular.py#L7)** , **[`build_state_names()`](pgmpy/utils/tabular.py#L12)**   (Utility Layer)**

The pandas-heaviest helpers. These are called by every discrete estimator. Migration starts here.
```python
# Before
def collect_state_names(data: pd.DataFrame, variable: str) -> list:
    return sorted(list(data.loc[:, variable].dropna().unique()))

# After
def collect_state_names(data: nw.DataFrame, variable: str) -> list:
    return sorted(data[variable].drop_nulls().unique().to_list())
```

**references**

**[`docs-"drop_nuls()"`](https://narwhals-dev.github.io/narwhals/api-reference/dataframe/#narwhals.dataframe.DataFrame.drop_nulls)**

**[`docs-"to_list()"`](https://narwhals-dev.github.io/narwhals/api-reference/series/#narwhals.series.Series.to_list)**

---

**Step 1b: **[`tabular.py`](pgmpy/utils/tabular.py)** - **[`get_state_counts()`](pgmpy/utils/tabular.py#L92)**   **(Utility Layer)**

The hardest function to migrate: uses `groupby(...).size().unstack()`, `pd.MultiIndex.from_product`, and `.reindex()`. The proposed approach is to route through the existing `get_state_counts_array()` (which already operates on integer codes + numpy), avoiding the pandas-heavy path entirely:

```python
# The narwhals migration keeps the fast numpy path via encode_columns + get_state_counts_array
# and only changes the input/output wrapping:

def get_state_counts(data, state_names, variable, parents=(), sample_weight=None, reindex=True):
    codes, cardinalities = encode_columns(data, state_names)
    counts_array = get_state_counts_array(codes, cardinalities, variable, parents, sample_weight)
    # Return as numpy array directly -- TabularCPD only needs np.array(state_counts)
    return counts_array
```

further we also have to update the **[`discrete_mle.py`](pgmpy\parameter_estimator\discrete_mle.py)**
as here: 
```python
    @staticmethod
    def _estimate_cpd(model, data, state_names: dict, node, sample_weight=None) -> TabularCPD:
        parents = sorted(model.get_parents(node))
        state_counts = get_state_counts(............)
        state_counts.iloc[:, (state_counts.values == 0).all(axis=0)] = 1.0
        ......
```

the `get_state_count()` used to use `iloc()` but after the update:
```python
# Before (pandas)
state_counts.iloc[:, (state_counts.values == 0).all(axis=0)] = 1.0

# After (numpy array)
zero_cols = (state_counts == 0).all(axis=0)
state_counts[:, zero_cols] = 1.0
```

---

**Step 1c: **[`unitls.py`](pgmpy/utils/utils.py)** - **[`preprocess_data()`](pgmpy/utils/utils.py#L293)**  (Utility Layer)**

Every discrete estimator calls this first. Heavy `pd.api.types.*` usage for dtype inference:

```python
# Before (pandas-only)
import pandas as pd

def preprocess_data(df):
    df = df.copy()
    dtypes = {}
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype("int")
            dtypes[col] = "N"
        elif pd.api.types.is_numeric_dtype(df[col]):
            dtypes[col] = "N"
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            dtypes[col] = "C"
            df[col] = df[col].astype("category")
        # ...
    return (df, dtypes)


# After (narwhals-compatible)
# No nw.from_native() needed here - _initialize_fit() already wraps the
# DataFrame before calling preprocess_data().
import narwhals as nw

def preprocess_data(df):
    dtypes = {}
    for col in df.columns:
        col_dtype = df[col].dtype
        if col_dtype.is_numeric():
            dtypes[col] = "N"
        elif col_dtype == nw.String or col_dtype == nw.Categorical:
            dtypes[col] = "C"
        elif isinstance(col_dtype, nw.Categorical) and col_dtype.is_ordered():
            dtypes[col] = "O"
        # ...
    return (df, dtypes)

# Modified (our change)
def _initialize_fit(self, model, data, sample_weight=None):
    data = nw.from_native(data) # updated and used narhwhals here
    data, _ = preprocess_data(data)
    ............

```

**Step 3: **[`discreate_mle.py`](pgmpy/parameter_estimator/discrete_mle.py)** - **[`fit()`](pgmpy/parameter_estimator/discrete_mle.py#L91)**, **[`_estimate_cpd()`](pgmpy/parameter_estimator/discrete_mle.py#L65)** (Pilot Target)**

Trace the full flow end-to-end: `fit()` --> `_initialize_fit()` --> `preprocess_data()` --> `build_state_names()` ->> per-node: `_estimate_cpd()` --> `get_state_counts()` --> `TabularCPD(np.array(state_counts))`. The pandas dataFrame becomes a numpy array at the `TabularCPD` boundary.

---

**Step 4: **[`power_divergence.py`](pgmpy/ci_tests/power_divergence.py)** - **[`__init__`](pgmpy/ci_tests/power_divergence.py#L130)** (CI Test Layer)**

Different pandas usage pattern than estimators - column selection and `pd.factorize()` for encoding. Narwhals doesn't expose `factorize` directly, but the same result can be achieved:

```python
# Before
for col in data.columns:
    codes, uniques = pd.factorize(data[col], sort=False, use_na_sentinel=True)
    self._codes[col] = np.ascontiguousarray(codes, dtype=np.int64)
    .........

# After
df = nw.from_native(data)
for col in df.columns:
    col_series = df[col]
    unique_vals = col_series.drop_nulls().unique().sort().to_list()
    val_to_code = {v: i for i, v in enumerate(unique_vals)}
    native_values = col_series.to_list()
    codes = np.array([val_to_code.get(v, -1) for v in native_values], dtype=np.int64)
    self._codes[col] = codes
    .........
```

---

> **NOTE**:
>  For performance-critical paths, we may keep `pd.factorize()` as an optimized specialization when the input is pandas, and use the narwhals path for other backends. This is a detail to decide during implementation.

**Step 5: **[`DiscreteBayesianNetwork.py`](pgmpy/models/DiscreteBayesianNetwork.py)** - [`fit()`](pgmpy/models/DiscreteBayesianNetwork.py#L602), **[`predict()`](pgmpy/models/DiscreteBayesianNetwork.py#L730)**, **[`simulate()`](pgmpy/models/DiscreteBayesianNetwork.py#L1393)** **(Model APIs)**

Highest-level public APIs accepting DataFrames. Check whether `fit()` just passes `data` through to an estimator (in which case Step 2 already handles the wrapping) or does its own pandas operations that also need migration.

---

#### Phased Rollout

| Phase | Scope | Key Files |
|-------|-------|-----------|
| **0. Pilot** | Single estimator end-to-end | `discrete_mle.py`, `base.py`, `tabular.py`, `utils.py` |
| **1. Parameter Estimators** | All discrete + Gaussian estimators | `discrete_bayesian.py`, `discrete_em.py`, `linear_gaussian_mle.py` |
| **2. CI Tests** | All conditional independence tests | `power_divergence.py`, `fisher_z.py`, `pearsonr.py`, `pillai_trace.py` |
| **3. Structure Scores** | All scoring functions | `BDeu`, `BDs`, `BIC`, `AIC`, `K2`, `LogLikelihood` |
| **4. Causal Discovery** | Search algorithms | `PC`, `GES`, `HillClimbSearch`, `ChowLiu` |
| **5. Model APIs** | Top-level model methods | `DiscreteBayesianNetwork.fit()`, `.predict()`, `.simulate()` |

Work for the phases will be done by seperate PRs.(this is divided into different phases as the pd.DataFrame is used in many source code files)


### User Journeys with the Solution
