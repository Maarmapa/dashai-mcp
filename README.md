# dashai-mcp

Servidor [MCP](https://modelcontextprotocol.io) para **[dashAI](https://github.com/DashAISoftware/DashAI)**, el workbench de Machine Learning open source de la Universidad de Chile (FCFM), desarrollado por estudiantes del DCC UChile y la UTFSM, con CENIA e IMFD.

Le da a un agente la misma superficie que dashAI le da a una persona por su interfaz gráfica: mirar datasets, ver qué modelos hay disponibles, entrenar, seguir el trabajo en cola y leer las métricas.

```
"Entrena un random forest sobre el dataset 3 prediciendo 'species' y dime el F1"
```

## Estado

**v0.2.0 — verificado contra dashAI 0.9.7.post1 corriendo.** 25 tests deterministas
en verde y un entrenamiento real completo de punta a punta: `dashai_train_model`
→ `dashai_job_status` → `dashai_get_run` con métricas.

La verificación contra una instancia viva encontró **cuatro errores que los tests
con dobles no podían ver**, todos por diferencias entre la documentación de dashAI
y su comportamiento real. Están descritos abajo y cada uno tiene su test de
regresión.

## Instalación

```bash
uv pip install git+https://github.com/Maarmapa/dashai-mcp
# o: pip install git+https://github.com/Maarmapa/dashai-mcp
```

Todavía no está publicado en PyPI.

En la configuración de tu cliente MCP:

```json
{
  "mcpServers": {
    "dashai": {
      "command": "dashai-mcp"
    }
  }
}
```

dashAI tiene que estar corriendo aparte (`dashai`, o la app de escritorio). Por defecto se busca en `http://localhost:8000`.

## Herramientas

| Herramienta | Qué hace |
|---|---|
| `dashai_server_info` | ¿Está dashAI arriba? Cuántos datasets y corridas hay |
| `dashai_list_datasets` | Lista los datasets cargados |
| `dashai_describe_dataset` | Columnas, tipos y muestra — todo en una llamada |
| `dashai_list_components` | Modelos, métricas, tareas y optimizadores disponibles |
| `dashai_train_model` | **Entrena.** Encola y devuelve `job_id` + `run_id` |
| `dashai_job_status` | Avance de un job: `not_started` / `started` / `finished` / `error` |
| `dashai_list_runs` | Corridas registradas, para comparar modelos |
| `dashai_get_run` | Configuración y métricas de una corrida |
| `dashai_predict` | Predice con el modelo de una corrida terminada |

## Cuatro cosas que la documentación de dashAI dice mal

Descubiertas corriéndolo contra una instancia real. Si escribes un cliente de esta
API, te van a morder:

| Lo que dice la doc | Lo que hace el código |
|---|---|
| `?select_types=["Model","Metric"]` | Deben ser **parámetros repetidos**: `?select_types=Model&select_types=Metric`. Con el array JSON responde 422. |
| `POST /job/` con cuerpo JSON | Es **form data**, con `kwargs` serializado como string JSON. Su propio `openapi.json` no declara `requestBody` para esa ruta, porque el endpoint parsea `request` a mano. |
| `splits` como objeto | Viaja como **string JSON**: el esquema Pydantic lo declara `str`. |
| `optimize(model_class, search_space, X, y, n_trials)` | La firma real es `optimize(model, input_dataset, output_dataset, parameters, metric)`, y `model` es una **instancia**, no una clase. |

Además, el registro de componentes tiene **13 tipos**, no los cuatro que sugiere
la documentación: `Task`, `GenerativeTask`, `Model`, `GenerativeModel`,
`DataLoader`, `DatasetSource`, `Metric`, `Optimizer`, `Job`, `LocalExplainer`,
`GlobalExplainer`, `Explorer`, `Converter`.

Y `GET /run/{id}` devuelve `split_indexes` con la lista completa de índices: en un
dataset de 10.000 filas son **59 KB, el 99% de la respuesta**. Este servidor la
reemplaza por el conteo por partición, dejando la respuesta en ~1 KB.

## Tres decisiones de diseño

### 1. Nueve herramientas, no 142

dashAI expone 142 endpoints REST. Generar una herramienta por endpoint es mecánico y es un error: un modelo con 140 herramientas gasta contexto leyendo el catálogo y elige peor. Estas nueve cubren el recorrido real de trabajo.

### 2. `dashai_train_model` colapsa tres llamadas

En la API cruda, entrenar es una secuencia encadenada:

```
POST /model-session/   → crea el experimento
POST /run/             → crea la corrida
POST /job/             → encola el ModelJob
```

Con campos obligatorios que la interfaz gráfica rellena sola y que no están documentados — `plot_history_path`, `plot_slice_path`, `plot_contour_path`, `plot_importance_path`. Además, **`splits` viaja como string JSON, no como objeto**, aunque la documentación de dashAI lo muestre como objeto: el esquema Pydantic del backend lo declara `str`. Ese tipo de detalle es exactamente lo que hace fallar a un agente contra la API cruda.

Acá es una sola llamada, y **no bloquea**: entrenar puede tomar horas, así que devuelve el `job_id` de inmediato y el avance se consulta con `dashai_job_status`.

### 3. Ninguna herramienta borra nada

La API de dashAI **no tiene autenticación** — se revisó endpoint por endpoint. Es coherente con algo local-first, pero significa que no hay ninguna barrera entre una frase mal interpretada y un `DELETE /dataset/{id}` irreversible. Borrar se hace desde la interfaz, mirando lo que se borra.

Por la misma razón, el servidor **se niega a apuntar a un host que no sea local**:

```
DASHAI_BASE_URL apunta a 'ml.ejemplo.com', que no es local, y la API de dashAI
no tiene autenticación: exponerla a la red deja el backend abierto a cualquiera
que lo alcance.
```

Se puede desactivar a propósito con `DASHAI_ALLOW_REMOTE=1`, si el destino está protegido por otra vía.

## Configuración

| Variable | Default | Para qué |
|---|---|---|
| `DASHAI_BASE_URL` | `http://localhost:8000` | Dónde está el backend |
| `DASHAI_ALLOW_REMOTE` | *(no)* | Permitir un host no local (ver arriba) |
| `DASHAI_TIMEOUT` | `30` | Segundos de espera por petición |

## Desarrollo

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q
```

Los tests simulan las respuestas HTTP con `respx`: **no necesitan una instancia de dashAI ni credenciales**. Prueban el contrato — qué llamadas se hacen, en qué orden, con qué cuerpo, y qué se le dice al agente cuando algo falla.

## Nota sobre el SDK

Requiere el SDK de Python de MCP **2.x**. La versión 2.0 eliminó `mcp.server.fastmcp`; ahora es `mcp.server.mcpserver.MCPServer` y las anotaciones son objetos `ToolAnnotations` en vez de diccionarios. La mayoría de los tutoriales todavía muestran la API 1.x.

## Licencia

MIT, igual que dashAI. Este es un servidor de terceros, **no oficial**: no está afiliado al proyecto dashAI ni a las instituciones que lo desarrollan.
