"""Testes rápidos das rotas (uso: .venv\\Scripts\\python smoke_test.py)."""

from fastapi.testclient import TestClient
from main import app

c = TestClient(app)


def f():
    return {"file": ("sample.h5", open("sample.h5", "rb"))}


r = c.post("/api/inspect", files=f())
print("INSPECT", r.status_code, len(r.json()["nodes"]), "nodes")

r = c.post("/api/convert", files=f())
print("CONVERT ALL (ZIP)", r.status_code, r.headers.get("content-disposition"), len(r.content), "bytes")

r = c.post("/api/convert", files=f(), data={"target_dataset": "/perfil/profundidade"})
print("CONVERT ONE (.RAW)", r.status_code, r.headers.get("content-disposition"), len(r.content), "bytes")
# profundidade tem 11 floats de 64 bits = 11 * 8 = 88 bytes
assert len(r.content) == 88, f"Esperado 88 bytes, recebido {len(r.content)}"
print("Validação de tamanho binário (.raw): SUCESSO (88 bytes exatos para float64[11])")

r = c.post("/api/convert", files=f(), data={"target_dataset": "/nao_existe"})
print("MISSING", r.status_code, r.text)

r = c.post("/api/inspect", files={"file": ("fake.h5", b"not hdf5")})
print("FAKE", r.status_code, r.text)

r = c.post("/api/inspect", files={"file": ("a.txt", b"x")})
print("BAD EXT", r.status_code, r.text)
