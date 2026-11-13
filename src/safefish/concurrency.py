"""Locks compartidos entre módulos."""

import threading

# Chroma carga el índice HNSW de forma perezosa en la primera consulta y esa carga no es segura
# entre hilos ("Error creating hnsw segment reader: Nothing found on disk" con el arnés en
# paralelo). Todas las consultas a Chroma pasan por este lock; cuestan milisegundos.
CHROMA_LOCK = threading.Lock()
