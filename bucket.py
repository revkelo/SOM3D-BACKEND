#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError

# ========= Cargar .env (opcional pero recomendado) =========
# pip install python-dotenv
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)
except Exception:
    pass  # si no está instalado, seguimos

# ========= Helpers para leer/sanear variables de entorno =========
def env(name: str, default: str | None = None):
    v = os.getenv(name)
    if v is None:
        return default
    # quita espacios laterales (y comentarios si el usuario los pegó al final)
    v = v.strip()
    if "#" in v:
        # si vino "1   # comentario", toma sólo antes del '#'
        v = v.split("#", 1)[0].strip()
    return v

def env_bool(name: str, default: bool = False) -> bool:
    v = env(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y", "on")

# ========= Configuración S3 / MinIO (usa tus variables) =========
AWS_ACCESS_KEY_ID = env("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env("AWS_SECRET_ACCESS_KEY")
AWS_REGION = env("AWS_REGION", "us-east-1")
S3_ENDPOINT = env("S3_ENDPOINT")  # None => AWS S3
S3_BUCKET = env("S3_BUCKET", "som3d")
S3_INSECURE = env_bool("S3_INSECURE", False)  # en HTTP no seguro para MinIO

IS_MINIO = S3_ENDPOINT is not None

session = boto3.session.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

common_cfg = Config(
    signature_version="s3v4",
    s3={"addressing_style": "path"} if IS_MINIO else {}
)

s3 = session.client(
    "s3",
    endpoint_url=S3_ENDPOINT,   # ej: http://127.0.0.1:9000 para MinIO; en AWS déjalo None
    config=common_cfg,
    verify=not S3_INSECURE,     # en MinIO http puedes setear S3_INSECURE=1
)

BUCKET = S3_BUCKET

# ========= Utilidades =========
def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024

def safe_run(fn):
    """Decorador para capturar errores y mostrarlos bonitos."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NoCredentialsError:
            messagebox.showerror(
                "Credenciales",
                "No se encontraron credenciales.\n\n"
                "Define AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY\n"
                "y opcionalmente S3_ENDPOINT (para MinIO)."
            )
        except EndpointConnectionError as e:
            messagebox.showerror("Endpoint", f"No se pudo conectar al endpoint.\n\n{e}")
        except ClientError as e:
            messagebox.showerror("S3 Error", f"{e.response.get('Error', {}).get('Code','')} - {e}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    return wrapper

def ensure_bucket_exists():
    """Asegura que el bucket exista; si no, lo crea (AWS o MinIO)."""
    try:
        s3.head_bucket(Bucket=BUCKET)
        return
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("404", "NotFound", "NoSuchBucket"):
            region = session.region_name or "us-east-1"
            if not IS_MINIO and region != "us-east-1":
                s3.create_bucket(
                    Bucket=BUCKET,
                    CreateBucketConfiguration={"LocationConstraint": region}
                )
            else:
                s3.create_bucket(Bucket=BUCKET)
        else:
            raise

# ========= Lógica S3 de alto nivel =========
def s3_list(prefix: str):
    """Genera objetos (Key, Size, LastModified) bajo prefix (o todo si prefix vacío)."""
    token = None
    while True:
        kwargs = {"Bucket": BUCKET, "Prefix": prefix or ""}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for item in resp.get("Contents", []):
            yield item["Key"], item["Size"], item.get("LastModified")
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break

def s3_upload(local_path: str, key: str, cb=None):
    s3.upload_file(local_path, BUCKET, key, Callback=cb)

def s3_download(key: str, local_path: str, cb=None):
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    s3.download_file(BUCKET, key, local_path, Callback=cb)

def s3_delete(key: str):
    s3.delete_object(Bucket=BUCKET, Key=key)

def s3_presigned_get(key: str, secs=3600):
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": key},
        ExpiresIn=secs
    )

# ========= GUI =========
class S3GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("S3/MinIO Uploader")
        self.geometry("950x600")
        self.minsize(900, 560)

        # Cola para logs desde hilos
        self.log_queue = queue.Queue()

        # Top: info y controles
        frm_top = ttk.Frame(self, padding=10)
        frm_top.pack(side=tk.TOP, fill=tk.X)

        endpoint_label = S3_ENDPOINT if S3_ENDPOINT else "(AWS S3)"
        ttk.Label(frm_top, text=f"Bucket: {BUCKET}").grid(row=0, column=0, sticky="w", padx=(0,12))
        ttk.Label(frm_top, text=f"Endpoint: {endpoint_label}").grid(row=0, column=1, sticky="w")

        ttk.Label(frm_top, text="Prefijo (carpeta):").grid(row=1, column=0, sticky="w")
        self.prefix_var = tk.StringVar(value="")
        self.ent_prefix = ttk.Entry(frm_top, textvariable=self.prefix_var, width=40)
        self.ent_prefix.grid(row=1, column=1, sticky="w", pady=5)

        self.btn_refresh = ttk.Button(frm_top, text="🔄 Refrescar", command=self.refresh_async)
        self.btn_refresh.grid(row=1, column=2, padx=5)

        # Middle: tabla objetos
        frm_mid = ttk.Frame(self, padding=(10, 0, 10, 5))
        frm_mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        cols = ("key", "size", "modified")
        self.tree = ttk.Treeview(frm_mid, columns=cols, show="headings", selectmode="extended")
        self.tree.heading("key", text="Key (ruta)")
        self.tree.heading("size", text="Tamaño")
        self.tree.heading("modified", text="Modificado")
        self.tree.column("key", width=520, anchor="w")
        self.tree.column("size", width=100, anchor="e")
        self.tree.column("modified", width=160, anchor="center")

        vsb = ttk.Scrollbar(frm_mid, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(frm_mid, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscroll=vsb.set, xscroll=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, columnspan=2, sticky="ew")

        frm_mid.rowconfigure(0, weight=1)
        frm_mid.columnconfigure(0, weight=1)

        # Bottom: acciones + progreso + log
        frm_bot = ttk.Frame(self, padding=10)
        frm_bot.pack(side=tk.BOTTOM, fill=tk.X)

        self.btn_select = ttk.Button(frm_bot, text="📁 Elegir archivo…", command=self.choose_file)
        self.btn_select.grid(row=0, column=0)

        self.sel_file_var = tk.StringVar(value="")
        ttk.Entry(frm_bot, textvariable=self.sel_file_var, width=50).grid(row=0, column=1, padx=6, sticky="we")

        self.btn_upload = ttk.Button(frm_bot, text="⬆ Subir", command=self.upload_async)
        self.btn_upload.grid(row=0, column=2, padx=4)

        self.btn_download = ttk.Button(frm_bot, text="⬇ Descargar", command=self.download_async)
        self.btn_download.grid(row=0, column=3, padx=4)

        self.btn_delete = ttk.Button(frm_bot, text="🗑 Borrar", command=self.delete_async)
        self.btn_delete.grid(row=0, column=4, padx=4)

        self.btn_url = ttk.Button(frm_bot, text="🔗 URL firmada", command=self.presign_selected)
        self.btn_url.grid(row=0, column=5, padx=4)

        self.prog = ttk.Progressbar(frm_bot, orient="horizontal", mode="determinate", length=250)
        self.prog.grid(row=1, column=0, columnspan=3, pady=(8,0), sticky="we")

        self.log_txt = tk.Text(frm_bot, height=6)
        self.log_txt.grid(row=2, column=0, columnspan=6, pady=(6,0), sticky="nsew")

        frm_bot.columnconfigure(1, weight=1)
        frm_bot.rowconfigure(2, weight=1)

        # Cargar lista inicial
        self.after(100, self.startup)

        # Loop para vaciar logs desde hilos
        self.after(150, self._drain_log_queue)

    @safe_run
    def startup(self):
        # Chequeos tempranos para mensajes claros
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
            raise NoCredentialsError()
        if IS_MINIO and not S3_ENDPOINT:
            raise EndpointConnectionError(endpoint_url="(vacío)")

        # Verifica que las credenciales funcionan (o al menos que el endpoint responde)
        try:
            s3.list_buckets()
        except NoCredentialsError:
            raise
        except EndpointConnectionError as e:
            raise
        except ClientError:
            # puede fallar en permisos; continuamos a head_bucket/create
            pass

        ensure_bucket_exists()
        self.refresh_async()

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_queue.put(f"[{ts}] {msg}\n")

    def _drain_log_queue(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                self.log_txt.insert("end", line)
                self.log_txt.see("end")
        except queue.Empty:
            pass
        finally:
            self.after(150, self._drain_log_queue)

    def choose_file(self):
        path = filedialog.askopenfilename(title="Selecciona un archivo")
        if path:
            self.sel_file_var.set(path)

    def _selected_keys(self):
        sel = []
        for item in self.tree.selection():
            values = self.tree.item(item, "values")
            if values:
                sel.append(values[0])
        return sel

    def _set_progress_bytes(self, total_size):
        # Callbacks de boto3 reciben bytes transferidos acumulados
        self.prog["maximum"] = total_size if total_size > 0 else 1
        def cb(bytes_amount):
            self.prog["value"] = bytes_amount
        return cb

    # ----- Acciones asíncronas -----
    def refresh_async(self):
        def work():
            self.log("Listando objetos…")
            prefix = self.prefix_var.get().strip()
            rows = []
            for key, size, lm in s3_list(prefix):
                mod = lm.strftime("%Y-%m-%d %H:%M:%S") if lm else ""
                rows.append((key, fmt_size(size), mod))
            # actualizar UI en hilo principal
            def update_tree():
                self.tree.delete(*self.tree.get_children())
                for r in rows:
                    self.tree.insert("", "end", values=r)
                self.log(f"{len(rows)} objeto(s) encontrados.")
            self.after(0, update_tree)
        threading.Thread(target=work, daemon=True).start()

    @safe_run
    def upload_async(self):
        local_path = self.sel_file_var.get().strip()
        if not local_path:
            messagebox.showwarning("Archivo", "Primero elige un archivo.")
            return
        if not os.path.isfile(local_path):
            messagebox.showerror("Archivo", "La ruta seleccionada no es un archivo.")
            return

        prefix = self.prefix_var.get().strip()
        key = f"{prefix}{os.path.basename(local_path)}" if prefix else os.path.basename(local_path)

        total_size = os.path.getsize(local_path)
        cb = self._set_progress_bytes(total_size)
        self.prog["value"] = 0

        def work():
            self.log(f"Subiendo: {local_path} → s3://{BUCKET}/{key}")
            try:
                s3_upload(local_path, key, cb=cb)
                self.log("✔ Subida completada.")
                self.after(0, self.refresh_async)
            except Exception as e:
                self.log(f"✖ Error de subida: {e}")
                messagebox.showerror("Upload", str(e))
            finally:
                self.after(0, lambda: self.prog.configure(value=0))
        threading.Thread(target=work, daemon=True).start()

    @safe_run
    def download_async(self):
        keys = self._selected_keys()
        if not keys:
            messagebox.showinfo("Descargar", "Selecciona al menos un objeto en la lista.")
            return
        folder = filedialog.askdirectory(title="Carpeta de destino para descargar")
        if not folder:
            return

        def work():
            for key in keys:
                # destino local: respeta subcarpetas del key
                local_path = os.path.join(folder, *key.split("/"))
                self.log(f"Descargando: s3://{BUCKET}/{key} → {local_path}")
                try:
                    # tamaño para barra
                    head = s3.head_object(Bucket=BUCKET, Key=key)
                    cb = self._set_progress_bytes(head.get("ContentLength", 0))
                    s3_download(key, local_path, cb=cb)
                    self.log("✔ Descarga completada.")
                except Exception as e:
                    self.log(f"✖ Error de descarga ({key}): {e}")
            self.after(0, lambda: self.prog.configure(value=0))
        threading.Thread(target=work, daemon=True).start()

    @safe_run
    def delete_async(self):
        keys = self._selected_keys()
        if not keys:
            messagebox.showinfo("Borrar", "Selecciona al menos un objeto.")
            return
        if not messagebox.askyesno("Confirmar", f"¿Borrar {len(keys)} objeto(s)?"):
            return

        def work():
            for key in keys:
                self.log(f"Borrando: s3://{BUCKET}/{key}")
                try:
                    s3_delete(key)
                except Exception as e:
                    self.log(f"✖ Error borrando {key}: {e}")
            self.after(0, self.refresh_async)
        threading.Thread(target=work, daemon=True).start()

    @safe_run
    def presign_selected(self):
        keys = self._selected_keys()
        if len(keys) != 1:
            messagebox.showinfo("URL firmada", "Selecciona exactamente un objeto.")
            return
        key = keys[0]
        url = s3_presigned_get(key, secs=3600)
        # Copiar al portapapeles y mostrar
        self.clipboard_clear()
        self.clipboard_append(url)
        parsed = urlparse(url)
        self.log(f"URL (copiada): {parsed.scheme}://{parsed.netloc}{parsed.path}?...")
        messagebox.showinfo("URL copiada", "URL firmada copiada al portapapeles (1h).")

if __name__ == "__main__":
    app = S3GUI()
    app.mainloop()
