"""Gera um arquivo HDF5 de exemplo para testar o conversor."""

import h5py
import numpy as np

with h5py.File("sample.h5", "w") as f:
    f.attrs["poco"] = "LC-PETRO-01"
    f.attrs["versao"] = 2

    g = f.create_group("perfil")
    d = g.create_dataset("profundidade", data=np.linspace(1000, 1010, 11))
    d.attrs["unidade"] = "m"
    g.create_dataset("gr", data=np.random.default_rng(0).normal(80, 10, (11, 3)))

    s = f.create_group("sensores/temperatura")
    s.create_dataset("serie", data=np.arange(24, dtype="int32").reshape(2, 3, 4))
    f.create_dataset("nomes", data=np.array(["alpha", "beta", "gama"], dtype=h5py.string_dtype()))
    f.create_dataset("escalar", data=3.14)

print("sample.h5 gerado")
