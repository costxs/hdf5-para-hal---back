"""
HDF5 -> RAW Converter API

Backend FastAPI + h5py.
- POST /api/inspect : lê a árvore (Groups/Datasets) do HDF5 e devolve metadados (shape, dtype, byte_size).
- POST /api/convert : converte o dataset escolhido (ou datasets numéricos) diretamente para binário puro (.raw).
  Se houver múltiplos datasets e nenhum for especificado, empacota todos em um .zip com arquivos .raw individuais.

Os uploads são gravados em disco via streaming e os datasets são lidos e gravados
em blocos binários para suportar arquivos volumosos sem estourar a memória RAM.
"""

import math
import os
import shutil
import tempfile
import zipfile
from typing import Any, BinaryIO, Dict, List, Tuple

import h5py
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

VALID_EXTENSIONS = (".h5", ".hdf5")
# Tamanho aproximado de cada bloco lido do dataset durante a escrita (8 MB)
CHUNK_TARGET_BYTES = 8 * 1024 * 1024

# Origens permitidas (separadas por vírgula).
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "*").split(",")
    if o.strip()
]

app = FastAPI(title="HDF5 to RAW Converter API")

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
        except Exception:
            out[k] = "<não legível>"
    return out


# --------------------------------------------------------------------------- #
# Escrita do formato Binário Puro (.raw)
# --------------------------------------------------------------------------- #
def write_dataset_raw_binary(raw_file: BinaryIO, ds: h5py.Dataset) -> int:
    """
    Grava os dados brutos (buffer binário C-contiguous) no arquivo .raw.
    Mantém o dtype original (ex.: float32, float64, int16, etc.) e endianness nativo.
    """
    if ds.shape is None or ds.size == 0:
        return 0

    if ds.dtype.kind in "OSU":
        raise HTTPException(
            status_code=400,
            detail=f"O dataset '{ds.name}' possui tipo '{ds.dtype}' que não é numérico para exportação binária pura .raw.",
        )

    bytes_written = 0
    if ds.ndim == 0:
        data = np.asarray(ds[()])
        b = data.tobytes()
        raw_file.write(b)
        return len(b)

    n_rows = ds.shape[0]
    row_items = int(np.prod(ds.shape[1:])) if ds.ndim > 1 else 1
    row_bytes = max(1, row_items * max(ds.dtype.itemsize, 1))
    rows_per_chunk = max(1, CHUNK_TARGET_BYTES // row_bytes)

    for start in range(0, n_rows, rows_per_chunk):
        stop = min(start + rows_per_chunk, n_rows)
        # Garante array contíguo em C (row-major) com o dtype nativo
        block = np.ascontiguousarray(ds[start:stop])
        b = block.tobytes()
        raw_file.write(b)
        bytes_written += len(b)

    return bytes_written


def convert_h5_to_raw_logic(
    h5_path: str,
    base_filename: str,
    target_dataset: str = "",
) -> Tuple[str, str, str]:
    """
    Exporta o dataset como binário puro (.raw).
    Se target_dataset for especificado, exporta esse dataset diretamente como .raw.
    Se não for especificado e houver apenas 1 dataset numérico, exporta como .raw.
    Se houver múltiplos datasets, empacota todos em um .zip com os arquivos .raw individuais.
    Retorna: (caminho_do_arquivo_gerado, nome_do_arquivo_para_download, media_type)
    """
    target = target_dataset.strip()

    with h5py.File(h5_path, "r") as h5_file:
        if target:
            key = "/" + target.lstrip("/")
            obj = h5_file.get(key)
            if not isinstance(obj, h5py.Dataset):
                raise HTTPException(status_code=404, detail=f"Dataset '{key}' não encontrado no arquivo.")

            clean_name = key.strip("/").replace("/", "_")
            out_filename = f"{base_filename}_{clean_name}.raw"
            tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".raw")
            output_path = tmp_out.name
            try:
                write_dataset_raw_binary(tmp_out, obj)
            finally:
                tmp_out.close()

            return output_path, out_filename, "application/octet-stream"

        # Nenhum dataset especificado: procurar todos os datasets numéricos
        dataset_paths: List[str] = []

        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind not in "OSU":
                dataset_paths.append("/" + name)

        h5_file.visititems(visitor)

        if not dataset_paths:
            raise HTTPException(
                status_code=400,
                detail="Nenhum dataset numérico encontrado para conversão binária .raw.",
            )

        if len(dataset_paths) == 1:
            key = dataset_paths[0]
            obj = h5_file[key]
            clean_name = key.strip("/").replace("/", "_")
            out_filename = f"{base_filename}_{clean_name}.raw"
            tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".raw")
            output_path = tmp_out.name
            try:
                write_dataset_raw_binary(tmp_out, obj)
            finally:
                tmp_out.close()
            return output_path, out_filename, "application/octet-stream"

        # Múltiplos datasets: criar ZIP contendo cada um como .raw
        tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        zip_path = tmp_zip.name
        tmp_zip.close()

        try:
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for key in dataset_paths:
                    obj = h5_file[key]
                    clean_name = key.strip("/").replace("/", "_")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".raw") as tmp_entry:
                        entry_path = tmp_entry.name
                    try:
                        with open(entry_path, "wb") as f_entry:
                            write_dataset_raw_binary(f_entry, obj)
                        zf.write(entry_path, arcname=f"{clean_name}.raw")
                    finally:
                        cleanup_temp_files(entry_path)
        except Exception:
            cleanup_temp_files(zip_path)
            raise

        out_filename = f"{base_filename}_datasets_raw.zip"
        return zip_path, out_filename, "application/zip"


# --------------------------------------------------------------------------- #
# Rotas
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/inspect")
def inspect_hdf5(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Lê a árvore do arquivo HDF5 e retorna seus Groups e Datasets com dimensões e tipo de dados."""
    filename = validate_extension(file.filename)
    tmp_path = save_upload_to_temp(file)

    nodes: List[Dict[str, Any]] = []
    try:
        with h5py.File(tmp_path, "r") as h5_file:

            def visitor(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset):
                    itemsize = int(obj.dtype.itemsize) if hasattr(obj.dtype, "itemsize") else 1
                    total_elements = int(obj.size) if obj.shape is not None else 0
                    nodes.append({
                        "path": f"/{name}",
                        "type": "dataset",
                        "shape": list(obj.shape) if obj.shape is not None else [],
                        "dtype": str(obj.dtype),
                        "size": total_elements,
                        "byte_size": total_elements * itemsize,
                        "attrs": len(obj.attrs),
                        "is_numeric": obj.dtype.kind not in "OSU",
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
def convert_hdf5_to_raw(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_dataset: str = Form(""),
) -> FileResponse:
    """Recebe o .hdf5, converte para binário puro .raw (ou .zip com os .raw) e devolve para download."""
    filename = validate_extension(file.filename)
    base_name = os.path.splitext(os.path.basename(filename))[0]

    input_path = save_upload_to_temp(file)
    output_path = None
    try:
        output_path, output_filename, media_type = convert_h5_to_raw_logic(
            input_path, base_name, target_dataset
        )
    except HTTPException:
        cleanup_temp_files(input_path)
        if output_path:
            cleanup_temp_files(output_path)
        raise
    except Exception as e:
        cleanup_temp_files(input_path)
        if output_path:
            cleanup_temp_files(output_path)
        raise HTTPException(status_code=500, detail=f"Falha na conversão para RAW: {e}")

    cleanup_temp_files(input_path)
    background_tasks.add_task(cleanup_temp_files, output_path)

    return FileResponse(
        path=output_path,
        filename=output_filename,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{output_filename}"'},
    )
