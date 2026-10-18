import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import cudf as cd
import modin.pandas as mpd
import warnings
warnings.filterwarnings("ignore")
import narwhals.stable.v2 as nw
from pgmpy.causal_discovery import ChowLiu
from pgmpy.datasets import load_dataset


# 1. Discrete Dataset (Biological network: 11 categorical variables)
df = load_dataset("sachs_discrete").data.sample(500, random_state=42)

# ========
# Pandas
# ========
est = ChowLiu()
est.fit(df)
print(f"Learned Edges(old):     {sorted(est.causal_graph_.edges())}")

# ========
# NumPy
# ========
numpy_array = df.to_numpy()
est_numpy = ChowLiuNew()
est_numpy.fit(numpy_array)
print(f"Learned Edges(numpy):   {sorted(est_numpy.causal_graph_.edges())}")

# ========
# Polars
# ========
polars_data = pl.DataFrame(df)
est_polars = ChowLiuNew()
est_polars.fit(nw.from_native(polars_data))
print(f"Learned Edges(polars):  {sorted(est_polars.causal_graph_.edges())}")

# ========
# Pyarrow
# ========
pyarrow_data = pa.Table.from_pandas(df, preserve_index=False)
est_pyarrow = ChowLiuNew()
est_pyarrow.fit(nw.from_native(pyarrow_data))
print(f"Learned Edges(pyarrow): {sorted(est_pyarrow.causal_graph_.edges())}")

# ========
# Modin
# ========
modin_data = mpd.DataFrame(df)
est_modin = ChowLiuNew()
est_modin.fit(nw.from_native(modin_data))
print(f"Learned Edges(modin):   {sorted(est_modin.causal_graph_.edges())}")

# ========
# cuDF
# ========
cudf_data = cd.DataFrame(df)
est_cudf = ChowLiuNew()
est_cudf.fit(nw.from_native(cudf_data))
print(f"Learned Edges(cudf):    {sorted(est_cudf.causal_graph_.edges())}")


# 2. Discrete Survey Dataset (Social survey: 5 categorical variables)
df = load_dataset("college_plans").data.sample(500, random_state=42)

# ========
# Pandas
# ========
est = ChowLiu()
est.fit(df)
print(f"Learned Edges(old):     {sorted(est.causal_graph_.edges())}")

# ========
# NumPy
# ========
numpy_array = df.to_numpy()
est_numpy = ChowLiuNew()
est_numpy.fit(numpy_array)
print(f"Learned Edges(numpy):   {sorted(est_numpy.causal_graph_.edges())}")

# ========
# Polars
# ========
polars_data = pl.DataFrame(df)
est_polars = ChowLiuNew()
est_polars.fit(nw.from_native(polars_data))
print(f"Learned Edges(polars):  {sorted(est_polars.causal_graph_.edges())}")

# ========
# Pyarrow
# ========
pyarrow_data = pa.Table.from_pandas(df, preserve_index=False)
est_pyarrow = ChowLiuNew()
est_pyarrow.fit(nw.from_native(pyarrow_data))
print(f"Learned Edges(pyarrow): {sorted(est_pyarrow.causal_graph_.edges())}")

# ========
# Modin
# ========
modin_data = mpd.DataFrame(df)
est_modin = ChowLiuNew()
est_modin.fit(nw.from_native(modin_data))
print(f"Learned Edges(modin):   {sorted(est_modin.causal_graph_.edges())}")

# ========
# cuDF
# ========
cudf_data = cd.DataFrame(df)
est_cudf = ChowLiuNew()
est_cudf.fit(nw.from_native(cudf_data))
print(f"Learned Edges(cudf):    {sorted(est_cudf.causal_graph_.edges())}")

# 3. Mixed Dataset (Categorical + Integer: 9 demographic variables)
df = load_dataset("contraceptive_method").data.dropna().sample(500, random_state=42)

# ========
# Pandas
# ========
est = ChowLiu()
est.fit(df)
print(f"Learned Edges(old):     {sorted(est.causal_graph_.edges())}")

# ========
# NumPy
# ========
numpy_array = df.to_numpy()
est_numpy = ChowLiuNew()
est_numpy.fit(numpy_array)
print(f"Learned Edges(numpy):   {sorted(est_numpy.causal_graph_.edges())}")

# ========
# Polars
# ========
polars_data = pl.DataFrame(df)
est_polars = ChowLiuNew()
est_polars.fit(nw.from_native(polars_data))
print(f"Learned Edges(polars):  {sorted(est_polars.causal_graph_.edges())}")

# ========
# Pyarrow
# ========
pyarrow_data = pa.Table.from_pandas(df, preserve_index=False)
est_pyarrow = ChowLiuNew()
est_pyarrow.fit(nw.from_native(pyarrow_data))
print(f"Learned Edges(pyarrow): {sorted(est_pyarrow.causal_graph_.edges())}")

# ========
# Modin
# ========
modin_data = mpd.DataFrame(df)
est_modin = ChowLiuNew()
est_modin.fit(nw.from_native(modin_data))
print(f"Learned Edges(modin):   {sorted(est_modin.causal_graph_.edges())}")

# ========
# cuDF
# ========
cudf_data = cd.DataFrame(df)
est_cudf = ChowLiuNew()
est_cudf.fit(nw.from_native(cudf_data))
print(f"Learned Edges(cudf):    {sorted(est_cudf.causal_graph_.edges())}")

# 4. Continuous Dataset (Flow cytometry: 11 float variables)
df = load_dataset("sachs_continuous").data.sample(500, random_state=42)

# ========
# Pandas
# ========
est = ChowLiu()
est.fit(df)
print(f"Learned Edges(old):     {sorted(est.causal_graph_.edges())}")

# ========
# NumPy
# ========
numpy_array = df.to_numpy()
est_numpy = ChowLiuNew()
est_numpy.fit(numpy_array)
print(f"Learned Edges(numpy):   {sorted(est_numpy.causal_graph_.edges())}")

# ========
# Polars
# ========
polars_data = pl.DataFrame(df)
est_polars = ChowLiuNew()
est_polars.fit(nw.from_native(polars_data))
print(f"Learned Edges(polars):  {sorted(est_polars.causal_graph_.edges())}")

# ========
# Pyarrow
# ========
pyarrow_data = pa.Table.from_pandas(df, preserve_index=False)
est_pyarrow = ChowLiuNew()
est_pyarrow.fit(nw.from_native(pyarrow_data))
print(f"Learned Edges(pyarrow): {sorted(est_pyarrow.causal_graph_.edges())}")

# ========
# Modin
# ========
modin_data = mpd.DataFrame(df)
est_modin = ChowLiuNew()
est_modin.fit(nw.from_native(modin_data))
print(f"Learned Edges(modin):   {sorted(est_modin.causal_graph_.edges())}")

# ========
# cuDF
# ========
cudf_data = cd.DataFrame(df)
est_cudf = ChowLiuNew()
est_cudf.fit(nw.from_native(cudf_data))
print(f"Learned Edges(cudf):    {sorted(est_cudf.causal_graph_.edges())}")
