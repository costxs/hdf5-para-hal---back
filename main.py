"""
HDF5 -> HAL Converter API

Backend FastAPI + h5py.
- POST /api/inspect : lê a árvore (Groups/Datasets) do HDF5 e devolve metadados.
- POST /api/convert : converte o HDF5 (todo ou um dataset específico) para .hal.

Os uploads são gravados em disco via streaming (sem carregar o arquivo inteiro
em memória) e os datasets são escritos no .hal em blocos, o que permite lidar
com arquivos grandes. Todos os temporários são removidos automaticamente.
"""

import math
import os
import shutil
import tempfile
from typing import Any, Dict, List, TextIO

import h5py
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

HAL_FORMAT_VERSION = "1.0"
VALID_EXTENSIONS = (".h5", ".hdf5")
# Tamanho aproximado de cada bloco lido do dataset durante a escrita (bytes)
CHUNK_TARGET_BYTES = 8 * 1024 * 1024

# Origens permitidas (separadas por vírgula). Em dev o Vite já faz proxy de /api,
# então o CORS só é necessário se o frontend for servido de outra origem.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "*").split(",")
    if o.strip()
]

app = FastAPI(title="HDF5 to HAL Converter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


# --------------------------------------------------------------------------- #
# Utilitários
# --------------------------------------------------------------------------- #
def cleanup_temp_files(*file_paths: str) -> None:
    """Remove os arquivos temporários após o envio da resposta."""
    for path in file_paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


def validate_extension(filename: str | None) -> str:
    if not filename or not filename.lower().endswith(VALID_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Formato inválido. Envie um arquivo .h5 ou .hdf5.")
    return filename


def save_upload_to_temp(file: UploadFile) -> str:
    """Grava o upload em disco por streaming e valida a assinatura HDF5."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        shutil.copyfileobj(file.file, tmp, length=1024 * 1024)
        tmp_path = tmp.name

    if not h5py.is_hdf5(tmp_path):
        cleanup_temp_files(tmp_path)
        raise HTTPException(status_code=400, detail="O arquivo enviado não é um HDF5 válido.")
    return tmp_path


def to_text(value: Any) -> str:
    """Converte valores do HDF5 (bytes, arrays, escalares NumPy) em texto legível."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return str([to_text(v) for v in value.tolist()]) if value.dtype.kind in "OSU" else str(value.tolist())
    if isinstance(value, np.generic):
        return to_text(value.item())
    if isinstance(value, (list, tuple)):
        return str([to_text(v) for v in value])
    return str(value)


def attrs_to_dict(attrs: h5py.AttributeManager) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k in attrs.keys():
        try:
            out[k] = to_text(attrs[k])
        except Exception:  # atributos com tipos exóticos não devem quebrar a leitura
            out[k] = "<não legível>"
    return out


# --------------------------------------------------------------------------- #
# Escrita do formato .hal
# --------------------------------------------------------------------------- #
def _write_rows(hal_file: TextIO, block: np.ndarray) -> None:
    """Escreve um bloco 2D (linhas x colunas) no arquivo .hal."""
    if block.dtype.kind in "biufc":
        # Caminho rápido para dados numéricos
        np.savetxt(hal_file, block, fmt="%s", delimiter="\t")
        return
    # Strings, bytes, objetos, tipos compostos etc.
    for row in block:
        hal_file.write("\t".join(to_text(v) for v in row) + "\n")


def write_dataset_data(hal_file: TextIO, ds: h5py.Dataset) -> None:
    """Escreve os dados do dataset em blocos, sem carregar tudo em memória."""
    if ds.shape is None or ds.ndim == 0:
        hal_file.write(to_text(ds[()]) + "\n")
        return
    if ds.size == 0:
        return

    source = ds.asstr() if h5py.check_string_dtype(ds.dtype) else ds

    n_rows = ds.shape[0]
    row_items = int(np.prod(ds.shape[1:])) if ds.ndim > 1 else 1
    row_bytes = max(1, row_items * max(ds.dtype.itemsize, 1))
    rows_per_chunk = max(1, CHUNK_TARGET_BYTES // row_bytes)
    cols = 1 if ds.ndim == 1 else ds.shape[-1]

    for start in range(0, n_rows, rows_per_chunk):
        stop = min(start + rows_per_chunk, n_rows)
        block = np.asarray(source[start:stop])
        # 1D -> uma coluna; ND -> achata dimensões iniciais em formato tabular
        block = block.reshape(-1, cols) if cols > 0 else block.reshape(-1, 1)
        _write_rows(hal_file, block)


def write_dataset_block(hal_file: TextIO, path: str, ds: h5py.Dataset) -> None:
    hal_file.write(f"[DATASET: {path}]\n")
    hal_file.write(f"SHAPE: {ds.shape}\n")
    hal_file.write(f"DTYPE: {ds.dtype}\n")
    for k, v in attrs_to_dict(ds.attrs).items():
        hal_file.write(f"ATTR_{k}: {v}\n")
    hal_file.write("DATA_START\n")
    write_dataset_data(hal_file, ds)
    hal_file.write("DATA_END\n\n")


def convert_h5_to_hal_logic(h5_path: str, hal_path: str, target_dataset: str = "") -> int:
    """
    Lógica central de leitura do HDF5 e escrita do formato .hal.
    Adapte a formatação de saída conforme a especificação exata do seu .hal.
    Retorna a quantidade de datasets exportados.
    """
    target = target_dataset.strip()

    with h5py.File(h5_path, "r") as h5_file, open(hal_path, "w", encoding="utf-8", newline="\n") as hal_file:
        # Cabeçalho padrão .hal
        hal_file.write(f"# HAL_FORMAT_VERSION: {HAL_FORMAT_VERSION}\n")
        hal_file.write(f"# SOURCE_ATTRIBUTES: {attrs_to_dict(h5_file.attrs)}\n\n")

        # Dataset específico: acesso direto, sem percorrer a árvore inteira
        if target:
            key = "/" + target.lstrip("/")
            obj = h5_file.get(key)
            if not isinstance(obj, h5py.Dataset):
                raise HTTPException(status_code=404, detail=f"Dataset '{key}' não encontrado no arquivo.")
            write_dataset_block(hal_file, key, obj)
            return 1

        count = 0

        def process_node(name: str, obj: Any) -> None:
            nonlocal count
            if isinstance(obj, h5py.Dataset):
                write_dataset_block(hal_file, f"/{name}", obj)
                count += 1

        h5_file.visititems(process_node)
        return count


# --------------------------------------------------------------------------- #
# Rotas
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# Rotas síncronas (def): o FastAPI as executa em threadpool, então o trabalho
# de I/O e CPU do h5py não bloqueia o event loop.
@app.post("/api/inspect")
def inspect_hdf5(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Lê a árvore do arquivo HDF5 e retorna seus Groups e Datasets."""
    filename = validate_extension(file.filename)
    tmp_path = save_upload_to_temp(file)

    nodes: List[Dict[str, Any]] = []
    try:
        with h5py.File(tmp_path, "r") as h5_file:

            def visitor(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset):
                    nodes.append({
                        "path": f"/{name}",
                        "type": "dataset",
                        "shape": list(obj.shape) if obj.shape is not None else [],
                        "dtype": str(obj.dtype),
                        "size": int(obj.size) if obj.shape is not None else 0,
                        "attrs": len(obj.attrs),
                    })
                elif isinstance(obj, h5py.Group):
                    nodes.append({"path": f"/{name}", "type": "group", "attrs": len(obj.attrs)})

            h5_file.visititems(visitor)
            root_attrs = attrs_to_dict(h5_file.attrs)

        return {
            "filename": filename,
            "size_bytes": os.path.getsize(tmp_path),
            "root_attrs": root_attrs,
            "nodes": nodes,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler estrutura HDF5: {e}")
    finally:
        cleanup_temp_files(tmp_path)


@app.post("/api/convert")
def convert_hdf5_to_hal(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_dataset: str = Form(""),
) -> FileResponse:
    """Recebe o .hdf5, executa a conversão e devolve o arquivo .hal para download."""
    filename = validate_extension(file.filename)
    output_filename = f"{os.path.splitext(os.path.basename(filename))[0]}.hal"

    input_path = save_upload_to_temp(file)
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".hal")
    output_path = tmp_out.name
    tmp_out.close()

    try:
        convert_h5_to_hal_logic(input_path, output_path, target_dataset)
    except HTTPException:
        cleanup_temp_files(input_path, output_path)
        raise
    except Exception as e:
        cleanup_temp_files(input_path, output_path)
        raise HTTPException(status_code=500, detail=f"Falha na conversão: {e}")

    # O HDF5 de entrada não é mais necessário; o .hal é removido após o download
    cleanup_temp_files(input_path)
    background_tasks.add_task(cleanup_temp_files, output_path)

    return FileResponse(
        path=output_path,
        filename=output_filename,
        media_type="application/octet-stream",
    )
