"""Testes rápidos das rotas (uso: .venv\\Scripts\\python smoke_test.py)."""

from fastapi.testclient import TestClient

from main import app

c = TestClient(app)


def f():
    return {"file": ("sample.h5", open("sample.h5", "rb"))}


r = c.post("/api/inspect", files=f())
print("INSPECT", r.status_code, r.json())

r = c.post("/api/convert", files=f())
print("CONVERT ALL", r.status_code, r.headers.get("content-disposition"))
print(r.text)

r = c.post("/api/convert", files=f(), data={"target_dataset": "/perfil/profundidade"})
print("CONVERT ONE", r.status_code)
print(r.text)

r = c.post("/api/convert", files=f(), data={"target_dataset": "/nao_existe"})
print("MISSING", r.status_code, r.text)

r = c.post("/api/inspect", files={"file": ("fake.h5", b"not hdf5")})
print("FAKE", r.status_code, r.text)

r = c.post("/api/inspect", files={"file": ("a.txt", b"x")})
print("BAD EXT", r.status_code, r.text)
