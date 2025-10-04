#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extrae información de PACIENTE desde un ZIP con DICOMs.

- Si no pasas la ruta del ZIP por CLI, abre un selector de archivos (Explorer) con tkinter.
- Lee solo metadatos (stop_before_pixels=True) → NO necesita códecs extra.
- Arregla tildes/ñ (mojibake) cuando el equipo no guardó SpecificCharacterSet.
- Por defecto imprime el primer paciente válido; con --all lista pacientes únicos.
- Salida en JSON.

Uso:
  python infodicom.py
  python infodicom.py "C:\\ruta\\estudio.zip"
  python infodicom.py --all
  python infodicom.py "C:\\ruta\\estudio.zip" --all
"""

import io
import json
import re
import sys
import zipfile
from typing import Dict, Optional, List, Tuple

from pydicom import dcmread, config as pydicom_config
from pydicom.errors import InvalidDicomError

# tolerar caracteres raros sin lanzar excepción
pydicom_config.encoding_errors = "ignore"

# ------------------- Utilidades de texto/encoding -------------------
def _looks_mojibake(s: str) -> bool:
    # Indicadores típicos de UTF-8 mal decodificado como latin-1
    return ("Ã" in s) or ("Â" in s) or ("�" in s)

def _rescue_latin1_utf8(s: str) -> str:
    """Convierte 'CaÃ±on' -> 'Cañon' si vino con mojibake."""
    try:
        return s.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return s

def _norm_spaces(s: Optional[str]) -> Optional[str]:
    if not s:
        return s
    return " ".join(s.split())

def _best_string(ds, tag_name: str) -> Optional[str]:
    """
    Devuelve el string del tag intentando decodificar correctamente
    cuando falta SpecificCharacterSet o está mal.
    """
    raw = getattr(ds, tag_name, None)
    if raw is None:
        return None

    s = str(raw).strip()
    if s and not _looks_mojibake(s):
        return _norm_spaces(s)

    # Intento 1: si no hay charset, asumir UTF-8 y re-decodificar
    try:
        if not getattr(ds, "SpecificCharacterSet", None):
            ds.SpecificCharacterSet = ["ISO_IR 192"]  # UTF-8
            ds.decode()  # pydicom re-decode all textual elements
            raw2 = getattr(ds, tag_name, None)
            s2 = str(raw2).strip() if raw2 is not None else None
            if s2 and not _looks_mojibake(s2):
                return _norm_spaces(s2)
    except Exception:
        pass

    # Intento 2: rescate latin1->utf8
    if s:
        return _norm_spaces(_rescue_latin1_utf8(s))

    return None

def _best_patient_name(ds) -> Optional[str]:
    """Especial para PN: reemplaza '^' por espacios y aplica rescates."""
    raw = getattr(ds, "PatientName", None)
    if raw is None:
        return None

    # 1) Lo que ya venga
    name = str(raw).strip()
    if name and not _looks_mojibake(name):
        return _norm_spaces(name.replace("^", " "))

    # 2) Re-decodificar asumiendo UTF-8 si no hay charset
    try:
        if not getattr(ds, "SpecificCharacterSet", None):
            ds.SpecificCharacterSet = ["ISO_IR 192"]
            ds.decode()
            raw2 = getattr(ds, "PatientName", None)
            name2 = str(raw2).strip() if raw2 is not None else None
            if name2 and not _looks_mojibake(name2):
                return _norm_spaces(name2.replace("^", " "))
    except Exception:
        pass

    # 3) Rescate latin1->utf8
    fixed = _rescue_latin1_utf8(name) if name else None
    return _norm_spaces((fixed or name or "").replace("^", " ")) or None

def _clean_value(v) -> Optional[str]:
    if v is None:
        return None
    try:
        s = str(v).strip()
        return s if s else None
    except Exception:
        return None

def _normalize_age(age: Optional[str]) -> Optional[str]:
    """
    Normaliza edad DICOM (AS).
    - Acepta '062Y', '02M', '7D', '003W', con posibles espacios/ minúsculas.
    - Quita ceros a la izquierda en el número.
    - Si unidad es 'Y', devuelve solo el número (sin 'Y').
    - Si unidad es M/W/D, conserva sufijo (p. ej., '3M', '2W', '7D').
    - Si no matchea, devuelve fallback sin espacios.
    """
    if not age:
        return None
    s = str(age).strip()
    m = re.search(r'(?i)(\d{1,3})\s*([YMWD])', s)
    if not m:
        return s.replace(" ", "")  # fallback

    n = int(m.group(1))          # elimina ceros a la izquierda
    u = m.group(2).upper()

    if u == "Y":
        return str(n)            # SIN 'Y'
    else:
        return f"{n}{u}"         # mantiene M/W/D

def _compact(d: Dict[str, Optional[str]]) -> Dict[str, str]:
    """Elimina claves con None o strings vacíos."""
    return {k: v for k, v in d.items() if v is not None and str(v).strip() != ""}

def _dataset_to_patient_dict(ds) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}

    # Nombre con lógica PN
    out["name"] = _best_patient_name(ds)

    # Otros campos con rescate de encoding
    out["id"]         = _best_string(ds, "PatientID")        or _clean_value(getattr(ds, "PatientID", None))
    out["sex"]        = _best_string(ds, "PatientSex")       or _clean_value(getattr(ds, "PatientSex", None))
    out["birth_date"] = _best_string(ds, "PatientBirthDate") or _clean_value(getattr(ds, "PatientBirthDate", None))

    raw_age = _best_string(ds, "PatientAge") or _clean_value(getattr(ds, "PatientAge", None))
    out["age"] = _normalize_age(raw_age) if raw_age else None

    # weight/size: solo incluir si tienen valor (y se compacta al final)
    w = _clean_value(getattr(ds, "PatientWeight", None))
    s = _clean_value(getattr(ds, "PatientSize", None))
    if w is not None:
        out["weight"] = w
    if s is not None:
        out["size"] = s

    # Devolver sin claves vacías
    return _compact(out)

# ------------------- Lógica principal ZIP -------------------
def _try_read_dicom_from_bytes(buf: bytes):
    bio = io.BytesIO(buf)
    return dcmread(bio, stop_before_pixels=True, force=True)

def first_patient_from_zip(zip_path: str) -> Tuple[Dict[str, Optional[str]], str]:
    """
    Devuelve (info_paciente, nombre_entrada_zip) del primer DICOM válido encontrado.
    Lanza RuntimeError si no se encuentra ninguno.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in sorted(zf.namelist()):
            if name.endswith("/") or name.endswith("\\"):
                continue  # directorio
            try:
                with zf.open(name, "r") as f:
                    data = f.read()
                ds = _try_read_dicom_from_bytes(data)
                if hasattr(ds, "PatientID") or hasattr(ds, "PatientName"):
                    return _dataset_to_patient_dict(ds), name
            except (InvalidDicomError, KeyError, zipfile.BadZipFile, RuntimeError, OSError):
                continue
    raise RuntimeError("No se encontró ningún DICOM válido con información de paciente en el ZIP.")

def all_unique_patients_from_zip(zip_path: str) -> List[Dict[str, Optional[str]]]:
    """
    Recorre todo el ZIP y devuelve pacientes únicos (clave: (PatientID, PatientName)).
    """
    seen = set()
    uniq: List[Dict[str, Optional[str]]] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/") or name.endswith("\\"):
                continue
            try:
                with zf.open(name, "r") as f:
                    data = f.read()
                ds = _try_read_dicom_from_bytes(data)
                info = _dataset_to_patient_dict(ds)
                key = ((info.get("id") or "").strip(), (info.get("name") or "").strip())
                if any(info.values()) and key not in seen:
                    seen.add(key)
                    uniq.append(info)
            except Exception:
                continue
    return uniq

# ------------------- Selector de archivos (Windows/desktop) -------------------
def pick_zip_with_tk_dialog() -> Optional[str]:
    """
    Abre un diálogo para seleccionar un archivo ZIP.
    Devuelve la ruta seleccionada o None si se cancela.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Selecciona un ZIP con DICOMs",
            filetypes=[("ZIP files", "*.zip"), ("Todos", "*.*")]
        )
        try:
            root.destroy()
        except Exception:
            pass
        return path if path else None
    except Exception:
        return None

# ------------------- CLI -------------------
def parse_args():
    import argparse
    ap = argparse.ArgumentParser(
        description="Extrae información de paciente desde un ZIP con DICOMs (con selector si no pasas ruta)."
    )
    ap.add_argument("zip", nargs="?", help="Ruta al archivo .zip con DICOMs")
    ap.add_argument("--all", action="store_true", help="Listar todos los pacientes únicos del ZIP")
    return ap.parse_args()

def main():
    args = parse_args()

    # Si no hay ruta, intentamos abrir selector de archivos
    zip_path = args.zip
    if not zip_path:
        zip_path = pick_zip_with_tk_dialog()
        if not zip_path:
            print(json.dumps({"error": "No se proporcionó ZIP y se canceló el selector."}, ensure_ascii=False, indent=2))
            sys.exit(2)

    try:
        if args.all:
            patients = all_unique_patients_from_zip(zip_path)
            print(json.dumps(
                {"zip": zip_path, "count": len(patients), "patients": patients},
                ensure_ascii=False, indent=2
            ))
        else:
            info, entry = first_patient_from_zip(zip_path)
            print(json.dumps(
                {"zip": zip_path, "zip_entry": entry, "patient": info},
                ensure_ascii=False, indent=2
            ))
    except Exception as e:
        print(json.dumps({"zip": zip_path, "error": str(e)}, ensure_ascii=False, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()
