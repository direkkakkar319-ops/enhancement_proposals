## Replace Hardcoded `pandas` Assumptions with `narwhals` for DataFrame-Agnostic Data Ingestion

Contributors: **[`@direkkakkar319`](https://github.com/direkkakkar319-ops)**


### Introduction

Related issues: **[`Issue 3395`](https://github.com/pgmpy/pgmpy/issues/3395)**.

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

### User Journeys with the Solution
