# SOM3D Backend con Docker

Este backend expone la API de procesamiento SOM3D empaquetada en contenedores Docker. El repositorio incluye una imagen multi-stage (GPU y CPU) y un `docker-compose.yml` listo para levantar el servicio con soporte para volumenes y GPUs Nvidia.

## Requisitos previos
- Docker Engine 24+ y el plugin Docker Compose v2.
- (GPU) Tarjeta Nvidia compatible, drivers recientes y Nvidia Container Toolkit instalados.
- Espacio en disco suficiente para almacenar los datasets en `./data`.

## Archivos relevantes
- `Dockerfile`: construye la imagen en dos variantes: `gpu` (por defecto, basada en `pytorch/pytorch` con CUDA 12.1) y `cpu` (basada en `python:3.10-slim`).
- `docker-compose.yml`: define el servicio `som3d`, mapea el puerto 8000 y monta `./data` dentro del contenedor.
- `requirements-common.txt`: dependencias de Python compartidas por ambas variantes.

## Primer arranque (GPU por defecto)
1. Crea la carpeta de datos si no existe: `mkdir data`.
2. Construye la imagen apuntando al stage GPU (ya configurado en compose):
   ```bash
   docker compose build
   ```
3. Arranca el servicio:
   ```bash
   docker compose up -d
   ```
4. La API quedara disponible en `http://localhost:8000`. La documentacion interactiva de FastAPI esta en `http://localhost:8000/docs`.
5. Sigue los logs en vivo si lo necesitas:
   ```bash
   docker compose logs -f som3d
   ```

El volumen `./data` queda montado en `/data` dentro del contenedor para almacenar archivos intermedios y resultados.

## Variante solo CPU
Si no tienes GPU disponible puedes cambiar la imagen objetivo al stage `cpu`:
1. Actualiza `docker-compose.yml` para apuntar al stage y tag CPU:
   ```yaml
   services:
     som3d:
       build:
         context: .
         target: cpu
       image: som3d:cpu
       # resto igual
   ```
2. Reconstruye y levanta el servicio como en la seccion anterior (`docker compose build && docker compose up -d`).

Otra opcion es construir manualmente la imagen CPU y luego ejecutar Compose:
```bash
docker build --target cpu -t som3d:cpu .
docker compose up -d --build
```

## Comandos utiles
- Detener el servicio y conservar datos: `docker compose down`.
- Detener y eliminar volumenes anonimos: `docker compose down -v`.
- Ejecutar un shell dentro del contenedor: `docker compose exec som3d bash`.
- Reconstruir forzando dependencias frescas: `docker compose build --no-cache`.

## Ciclo de desarrollo
- Modifica el codigo localmente y vuelve a ejecutar `docker compose up --build` para aplicar cambios.
- Usa `docker compose restart som3d` cuando solo necesites reiniciar el proceso sin reconstruir.

## Salud y diagnostico
- Revisa `http://localhost:8000/health` (si esta ruta existe en la API).
- Verifica que las GPUs esten visibles: `docker compose exec som3d python -c "import torch; print(torch.cuda.is_available())"`.
- Asegurate de que `./data` tenga el espacio suficiente antes de correr trabajos pesados.

## Limpieza
Para eliminar la imagen creada manualmente: `docker image rm som3d:gpu` (o `som3d:cpu`). Usa `docker system prune` para liberar cache de compilaciones si lo necesitas.


docker compose up --build -d
