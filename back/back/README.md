# Crear virtual environment

```bash
python3.11 -m venv apivenv
```

# Activar virtual environment

```bash
source apivenv/bin/activate
```

# Instalar dependencias

```bash
pip3.11 install -r requirements.txt
```

# Iniciar el servidor

```bash
uvicorn main:app --reload
```
